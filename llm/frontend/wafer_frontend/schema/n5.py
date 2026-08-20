"""Versioned N5 profile-complete projection and scheduling wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .action import FusionPlan, StandaloneCollectivePlan
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import GlobalActionDAG
from .ir1 import IR1
from .ir0 import (
    CrossEntropyBackwardWorkload,
    CrossEntropyForwardWorkload,
    OpKind,
    OpPhase,
    SgdUpdateWorkload,
    StateAccessMode,
    TrainStructure,
)
from .ir2 import (
    BufferAccess,
    BufferUseRole,
    IR2ProjectionResult,
    IntraDieScheduleSet,
    OrdinaryNodeOrigin,
    SemanticTaskKind,
    StateIoOrigin,
)
from .logical import ProfileEntry
from .n4 import (
    InterDiePlanBundle,
    InterDiePlannedProfile,
    Stage4InterDiePlannedIR1,
    TrainInterDiePlannedIR1,
    TrainReplicaInterDiePlans,
)
from .persistent_state import PersistentStateAccess, StateKind
from .policy import PolicySelection, RegistryKind
from .stage4_pd import Stage4PdMode, Stage4PdPlan
from .state_transfer import (
    SlicedKvStateTransferContract,
    StateTransferContract,
    StateTransferLike,
)

if TYPE_CHECKING:
    from ..lowering.context import LoweringContext


PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.project_to_ir2_context/v1alpha6"
)
STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.stage4_project_to_ir2_context/v1alpha1"
)
INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_scheduling_context/v1alpha7"
)
PROJECTED_IR2_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.projected_ir2_bundle/v1alpha6"
)
STAGE4_PROJECTED_IR2_SCHEMA_VERSION = (
    "wafer_frontend.stage4_projected_ir2/v1alpha2"
)
TRAIN_PROJECTED_IR2_SCHEMA_VERSION = (
    "wafer_frontend.train_projected_ir2/v1alpha2"
)
STAGE4_SCHEDULED_IR2_SCHEMA_VERSION = (
    "wafer_frontend.stage4_scheduled_ir2/v1alpha2"
)
TRAIN_SCHEDULED_IR2_SCHEMA_VERSION = (
    "wafer_frontend.train_scheduled_ir2/v1alpha2"
)
SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.scheduled_ir2_bundle/v1alpha6"
)
GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.global_action_bundle/v1alpha6"
)
STAGE4_GLOBAL_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.stage4_global_action/v1alpha2"
)


class ProjectToIR2Contract(str, Enum):
    EXACT_NAIVE_PROJECTION_STATE_TRANSFER_V6 = (
        "exact_naive_projection_state_transfer/v6"
    )
    EXACT_NAIVE_PROJECTION_DENSE_FORWARD_V5 = (
        "exact_naive_projection_dense_forward/v5"
    )


class IntraDieSchedulingContract(str, Enum):
    NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5 = (
        "naive_component_rr_xy_sequential_state_transfer/v5"
    )


@dataclass(frozen=True, slots=True)
class ProjectToIR2Context:
    schema_version: str
    producer_pass: str
    id: str
    contract: ProjectToIR2Contract
    state_transfers: tuple[StateTransferLike, ...]

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        state_transfers: tuple[StateTransferLike, ...],
        contract: ProjectToIR2Contract = (
            ProjectToIR2Contract.EXACT_NAIVE_PROJECTION_STATE_TRANSFER_V6
        ),
    ) -> "ProjectToIR2Context":
        semantic_key = {
            "contract": contract,
            "state_transfers": state_transfers,
        }
        return cls(
            schema_version=PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "project_to_ir2_context",
                semantic_key,
                schema_version=PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str = "project_to_ir2_context") -> None:
        if self.schema_version != PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.contract) is not ProjectToIR2Contract:
            raise SchemaError(
                "must be a ProjectToIR2Contract",
                path=f"{path}.contract",
            )
        if (
            self.contract
            is not ProjectToIR2Contract.EXACT_NAIVE_PROJECTION_STATE_TRANSFER_V6
        ):
            raise SchemaError(
                "unsupported projection contract",
                path=f"{path}.contract",
            )
        if type(self.state_transfers) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.state_transfers"
            )
        transfer_ids: set[str] = set()
        source_keys: set[tuple[str, str]] = set()
        destination_keys: set[tuple[str, str]] = set()
        transfer_types_by_ir1: dict[str, type[object]] = {}
        for index, transfer in enumerate(self.state_transfers):
            transfer_path = f"{path}.state_transfers[{index}]"
            if type(transfer) not in (
                StateTransferContract,
                SlicedKvStateTransferContract,
            ):
                raise SchemaError(
                    "must be a StateTransferLike", path=transfer_path
                )
            transfer.validate(transfer_path)
            previous_type = transfer_types_by_ir1.setdefault(
                transfer.source_ir1_id, type(transfer)
            )
            if previous_type is not type(transfer):
                raise SchemaError(
                    "one IR-1 cannot mix whole and sliced state transfers",
                    path=transfer_path,
                )
            source_key = (
                transfer.source_ir1_id,
                transfer.source_state_access_ref,
            )
            destination_key = (
                transfer.source_ir1_id,
                transfer.destination_state_access_ref,
            )
            duplicate_endpoint = (
                type(transfer) is StateTransferContract
                and (
                    source_key in source_keys
                    or destination_key in destination_keys
                )
            )
            if transfer.id in transfer_ids or duplicate_endpoint:
                raise SchemaError(
                    "duplicate transfer id or whole-state endpoint access",
                    path=transfer_path,
                )
            transfer_ids.add(transfer.id)
            source_keys.add(source_key)
            destination_keys.add(destination_key)
        expected_id = stable_artifact_id(
            "project_to_ir2_context",
            {
                "contract": self.contract,
                "state_transfers": self.state_transfers,
            },
            schema_version=PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class Stage4ProjectToIR2Context:
    """Stage 4 projection context with no caller-supplied transfer payload."""

    schema_version: str
    producer_pass: str
    id: str
    contract: ProjectToIR2Contract

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        contract: ProjectToIR2Contract = (
            ProjectToIR2Contract.EXACT_NAIVE_PROJECTION_STATE_TRANSFER_V6
        ),
    ) -> "Stage4ProjectToIR2Context":
        semantic_key = {"contract": contract}
        return cls(
            schema_version=STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "stage4_project_to_ir2_context",
                semantic_key,
                schema_version=(
                    STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION
                ),
            ),
            **semantic_key,
        )

    def validate(
        self,
        path: str = "stage4_project_to_ir2_context",
    ) -> None:
        if (
            self.schema_version
            != STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.contract) is not ProjectToIR2Contract:
            raise SchemaError(
                "must be a ProjectToIR2Contract",
                path=f"{path}.contract",
            )
        if (
            self.contract
            is not ProjectToIR2Contract
            .EXACT_NAIVE_PROJECTION_STATE_TRANSFER_V6
        ):
            raise SchemaError(
                "unsupported Stage 4 projection contract",
                path=f"{path}.contract",
            )
        expected_id = stable_artifact_id(
            "stage4_project_to_ir2_context",
            {"contract": self.contract},
            schema_version=STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class IntraDieSchedulingContext:
    schema_version: str
    producer_pass: str
    id: str
    policy: PolicySelection
    contract: IntraDieSchedulingContract

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        policy: PolicySelection,
        contract: IntraDieSchedulingContract = (
            IntraDieSchedulingContract
            .NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5
        ),
    ) -> "IntraDieSchedulingContext":
        semantic_key = {"policy": policy, "contract": contract}
        return cls(
            schema_version=INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "intra_die_scheduling_context",
                semantic_key,
                schema_version=INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str = "intra_die_scheduling_context") -> None:
        if self.schema_version != INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.policy) is not PolicySelection:
            raise SchemaError(
                "must be a PolicySelection", path=f"{path}.policy"
            )
        self.policy.validate(f"{path}.policy")
        if (
            self.policy.kind is not RegistryKind.INTRA_DIE
            or self.policy.name != "naive"
        ):
            raise SchemaError(
                "must select the registered naive intra-die policy",
                path=f"{path}.policy",
            )
        if type(self.contract) is not IntraDieSchedulingContract:
            raise SchemaError(
                "must be an IntraDieSchedulingContract",
                path=f"{path}.contract",
            )
        if (
            self.contract
            is not IntraDieSchedulingContract
            .NAIVE_COMPONENT_RR_XY_SEQUENTIAL_STATE_TRANSFER_V5
        ):
            raise SchemaError(
                "unsupported scheduling contract",
                path=f"{path}.contract",
            )
        expected_id = stable_artifact_id(
            "intra_die_scheduling_context",
            {"policy": self.policy, "contract": self.contract},
            schema_version=INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


def _validate_profile_manifest(
    profiles: tuple[ProfileEntry, ...],
    *,
    path: str,
) -> None:
    if type(profiles) is not tuple or not profiles:
        raise SchemaError(
            "must be a non-empty immutable tuple",
            path=path,
        )
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


def _validate_weight(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SchemaError("must be a finite positive float", path=path)


def _validate_projected_payload(
    *,
    graph: IR1,
    fusion_plans: tuple[FusionPlan, ...],
    standalone_plans: tuple[StandaloneCollectivePlan, ...],
    projection: IR2ProjectionResult,
    profile_id: str,
    path: str,
) -> None:
    if type(graph) is not IR1:
        raise SchemaError("must be an IR1", path=f"{path}.graph")
    graph.validate(f"{path}.graph")
    if graph.producer_pass != "fusion_partition":
        raise SchemaError(
            "must embed fusion_partition IR1",
            path=f"{path}.graph.producer_pass",
        )
    if profile_id != graph.profile.stable_id():
        raise SchemaError(
            "must equal graph profile id",
            path=f"{path}.profile_id",
        )
    if type(fusion_plans) is not tuple or type(standalone_plans) is not tuple:
        raise SchemaError(
            "plan collections must be immutable tuples",
            path=path,
        )
    if type(projection) is not IR2ProjectionResult:
        raise SchemaError(
            "must be an IR2ProjectionResult",
            path=f"{path}.projection",
        )
    if projection.producer_pass != "project_to_ir2":
        raise SchemaError(
            "must be produced by project_to_ir2",
            path=f"{path}.projection.producer_pass",
        )
    projection.validate_against(
        graph,
        fusion_plans,
        standalone_plans,
        f"{path}.projection",
    )


@dataclass(frozen=True, slots=True)
class ProjectedProfileIR2:
    id: str
    source_planned_entry_id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    projection_context_id: str
    profile_id: str
    weight: float
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult

    @classmethod
    def create(
        cls,
        *,
        source: InterDiePlannedProfile,
        context: ProjectToIR2Context,
        projection: IR2ProjectionResult,
    ) -> "ProjectedProfileIR2":
        semantic_key = {
            "source_planned_entry_id": source.id,
            "source_partitioned_entry_id": source.source_partitioned_entry_id,
            "source_ir1_id": source.graph.id,
            "projection_context_id": context.id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in source.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": projection.id,
        }
        return cls(
            id=stable_artifact_id(
                "projected_profile_ir2",
                semantic_key,
                schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
            source_planned_entry_id=source.id,
            source_partitioned_entry_id=source.source_partitioned_entry_id,
            source_ir1_id=source.graph.id,
            projection_context_id=context.id,
            profile_id=source.profile_id,
            weight=source.weight,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=projection,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_planned_entry_id": self.source_planned_entry_id,
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "projection_context_id": self.projection_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
        }

    def validate(self, path: str = "projected_profile_ir2") -> None:
        for field_name in (
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
            "profile_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_weight(self.weight, f"{path}.weight")
        _validate_projected_payload(
            graph=self.graph,
            fusion_plans=self.fusion_plans,
            standalone_plans=self.standalone_plans,
            projection=self.projection,
            profile_id=self.profile_id,
            path=path,
        )
        if self.source_ir1_id != self.graph.id:
            raise SchemaError(
                "must equal embedded graph id",
                path=f"{path}.source_ir1_id",
            )
        expected_id = stable_artifact_id(
            "projected_profile_ir2",
            self._semantic_key(),
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: InterDiePlannedProfile,
        context: ProjectToIR2Context,
        path: str = "projected_profile_ir2",
    ) -> None:
        source.validate("inter_die_planned_entry")
        context.validate("project_to_ir2_context")
        self.validate(path)
        if self.source_planned_entry_id != source.id:
            raise SchemaError(
                "must match source planned entry",
                path=f"{path}.source_planned_entry_id",
            )
        if self.source_partitioned_entry_id != source.source_partitioned_entry_id:
            raise SchemaError(
                "must preserve source partitioned entry",
                path=f"{path}.source_partitioned_entry_id",
            )
        if self.projection_context_id != context.id:
            raise SchemaError(
                "must match projection context",
                path=f"{path}.projection_context_id",
            )
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError("must preserve source profile and weight", path=path)
        if (
            self.source_ir1_id != source.graph.id
            or self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
        ):
            raise SchemaError(
                "must preserve the complete planned payload",
                path=path,
            )
        expected_transfers = tuple(
            transfer
            for transfer in context.state_transfers
            if transfer.source_ir1_id == source.graph.id
        )
        if self.projection.state_transfers != expected_transfers:
            raise SchemaError(
                "projection must exactly preserve context state transfers",
                path=f"{path}.projection.state_transfers",
            )


@dataclass(frozen=True, slots=True)
class ProjectedIR2Bundle:
    schema_version: str
    producer_pass: str
    id: str
    source_inter_die_bundle_id: str
    source_partitioned_bundle_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[ProjectedProfileIR2, ...]

    @classmethod
    def create(
        cls,
        *,
        source: InterDiePlanBundle,
        context: ProjectToIR2Context,
        entries: tuple[ProjectedProfileIR2, ...],
    ) -> "ProjectedIR2Bundle":
        semantic_key = {
            "source_inter_die_bundle_id": source.id,
            "source_partitioned_bundle_id": source.source_partitioned_bundle_id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": context.id,
            "source_profiles": source.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
            producer_pass="project_to_ir2",
            id=stable_artifact_id(
                "projected_ir2_bundle",
                semantic_key,
                schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_inter_die_bundle_id": self.source_inter_die_bundle_id,
            "source_partitioned_bundle_id": self.source_partitioned_bundle_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "projected_ir2_bundle") -> None:
        if self.schema_version != PROJECTED_IR2_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "project_to_ir2":
            raise SchemaError(
                "must be 'project_to_ir2'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_profile_manifest(
            self.source_profiles,
            path=f"{path}.source_profiles",
        )
        if type(self.entries) is not tuple or len(self.entries) != len(
            self.source_profiles
        ):
            raise SchemaError(
                "must contain one immutable entry per profile",
                path=f"{path}.entries",
            )
        source_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not ProjectedProfileIR2:
                raise SchemaError(
                    "must be a ProjectedProfileIR2",
                    path=entry_path,
                )
            entry.validate(entry_path)
            if (
                entry.profile_id != profile.profile_id
                or entry.graph.profile != profile.key
            ):
                raise SchemaError(
                    "must match source profile order",
                    path=f"{entry_path}.profile_id",
                )
            if entry.weight != profile.weight:
                raise SchemaError(
                    "must match source profile weight",
                    path=f"{entry_path}.weight",
                )
            if entry.projection_context_id != self.projection_context_id:
                raise SchemaError(
                    "must match bundle projection context",
                    path=f"{entry_path}.projection_context_id",
                )
            if entry.source_planned_entry_id in source_ids:
                raise SchemaError(
                    "duplicate source planned entry",
                    path=f"{entry_path}.source_planned_entry_id",
                )
            source_ids.add(entry.source_planned_entry_id)
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
                    "all profiles must share fabric, groups, and fusion partition",
                    path=f"{entry_path}.graph",
                )
        expected_id = stable_artifact_id(
            "projected_ir2_bundle",
            self._semantic_key(),
            schema_version=PROJECTED_IR2_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: InterDiePlanBundle,
        context: ProjectToIR2Context,
        path: str = "projected_ir2_bundle",
    ) -> None:
        source.validate("inter_die_plan_bundle")
        context.validate("project_to_ir2_context")
        self.validate(path)
        if self.source_inter_die_bundle_id != source.id:
            raise SchemaError(
                "must match source inter-die bundle",
                path=f"{path}.source_inter_die_bundle_id",
            )
        if self.source_partitioned_bundle_id != source.source_partitioned_bundle_id:
            raise SchemaError(
                "must preserve source partitioned bundle",
                path=f"{path}.source_partitioned_bundle_id",
            )
        if (
            self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
            or self.planning_context_id != source.planning_context_id
        ):
            raise SchemaError("must preserve upstream contexts", path=path)
        if self.projection_context_id != context.id:
            raise SchemaError(
                "must match projection context",
                path=f"{path}.projection_context_id",
            )
        source_ir1_ids = {entry.graph.id for entry in source.entries}
        if any(
            transfer.source_ir1_id not in source_ir1_ids
            for transfer in context.state_transfers
        ):
            raise SchemaError(
                "projection context contains an orphan state transfer",
                path="project_to_ir2_context.state_transfers",
            )
        if (
            self.source_profiles != source.source_profiles
            or len(self.entries) != len(source.entries)
        ):
            raise SchemaError(
                "must preserve complete profile manifest",
                path=f"{path}.source_profiles",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry.validate_against(
                source_entry,
                context,
                f"{path}.entries[{index}]",
            )


def _is_s2_lite_train_graph(graph: IR1) -> bool:
    kinds = {node.kind for node in graph.nodes}
    return (
        OpKind.CE_BACKWARD in kinds
        or OpKind.OPTIMIZER_UPDATE in kinds
        or any(node.phase is OpPhase.WGRAD for node in graph.nodes)
    )


def _validate_s2_lite_projection_state(
    graph: IR1,
    projection: IR2ProjectionResult,
    path: str,
) -> None:
    """Close the one-trainable-parameter Lite state contract exactly."""

    manifest = graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError(
            "S2-Lite projection requires persistent state",
            path=f"{path}.graph.persistent_state_manifest",
        )
    trainable = tuple(
        declaration
        for declaration in manifest.declarations
        if declaration.identity.kind is StateKind.TRAINABLE_PARAMETER
    )
    frozen = tuple(
        declaration
        for declaration in manifest.declarations
        if declaration.identity.kind is StateKind.PARAMETER
    )
    if (
        len(trainable) != 1
        or len(frozen) + len(trainable) != len(manifest.declarations)
        or trainable[0].access is not PersistentStateAccess.READ_WRITE
        or any(
            declaration.access is not PersistentStateAccess.READ_ONLY
            for declaration in frozen
        )
    ):
        raise SchemaError(
            "S2-Lite requires one READ_WRITE trainable parameter and READ_ONLY frozen parameters",
            path=f"{path}.graph.persistent_state_manifest.declarations",
        )

    node_index = {node.id: node for node in graph.nodes}
    trainable_accesses = tuple(
        access
        for access in graph.state_accesses
        if access.state_ref == trainable[0].id
    )
    if (
        len(trainable_accesses) != 2
        or tuple(access.mode for access in trainable_accesses)
        != (StateAccessMode.READ, StateAccessMode.READ_WRITE)
        or node_index[trainable_accesses[0].node_ref].kind is not OpKind.GEMM
        or node_index[trainable_accesses[1].node_ref].kind
        is not OpKind.OPTIMIZER_UPDATE
    ):
        raise SchemaError(
            "S2-Lite trainable state must be read by LM head and updated once",
            path=f"{path}.graph.state_accesses",
        )
    ce_backward = tuple(
        node for node in graph.nodes if node.kind is OpKind.CE_BACKWARD
    )
    updates = tuple(
        node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE
    )
    wgrads = tuple(
        node
        for node in graph.nodes
        if node.kind is OpKind.GEMM and node.phase is OpPhase.WGRAD
    )
    if (
        len(ce_backward) != 1
        or type(ce_backward[0].workload) is not CrossEntropyBackwardWorkload
        or len(wgrads) != 1
        or len(updates) != 1
        or type(updates[0].workload) is not SgdUpdateWorkload
    ):
        raise SchemaError(
            "S2-Lite requires exact CE_BACKWARD, LM-head WGRAD, and SGD update nodes",
            path=f"{path}.graph.nodes",
        )

    dma_out = tuple(
        task
        for dag in projection.dags
        for task in dag.tasks
        if task.kind is SemanticTaskKind.DMA_OUT
    )
    if (
        len(dma_out) != 1
        or not isinstance(dma_out[0].origin_ref, StateIoOrigin)
        or dma_out[0].origin_ref.state_access_ref != trainable_accesses[1].id
    ):
        raise SchemaError(
            "S2-Lite projection requires one exact trainable-parameter DMA_OUT",
            path=f"{path}.projection.dags",
        )


@dataclass(frozen=True, slots=True)
class TrainProjectedReplica:
    id: str
    replica_index: int
    source_replica_plan_id: str
    source_ir1_id: str
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult

    @classmethod
    def create(
        cls,
        *,
        source: TrainReplicaInterDiePlans,
        projection: IR2ProjectionResult,
    ) -> "TrainProjectedReplica":
        semantic_key = {
            "replica_index": source.replica_index,
            "source_replica_plan_id": source.id,
            "source_ir1_id": source.graph.id,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in source.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": projection.id,
        }
        return cls(
            id=stable_artifact_id(
                "train_projected_replica",
                semantic_key,
                schema_version=TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
            ),
            replica_index=source.replica_index,
            source_replica_plan_id=source.id,
            source_ir1_id=source.graph.id,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=projection,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "replica_index": self.replica_index,
            "source_replica_plan_id": self.source_replica_plan_id,
            "source_ir1_id": self.source_ir1_id,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
        }

    def validate(self, path: str) -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        validate_nonempty(
            self.source_replica_plan_id,
            f"{path}.source_replica_plan_id",
        )
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if (
            self.source_ir1_id != self.graph.id
            or self.graph.producer_pass != "fusion_partition"
            or len(self.graph.groups) != 1
            or not self.graph.groups[0].id.endswith(f"__dp{self.replica_index}")
        ):
            raise SchemaError(
                "must embed the canonical replica fusion IR1",
                path=f"{path}.graph",
            )
        if type(self.fusion_plans) is not tuple or type(
            self.standalone_plans
        ) is not tuple:
            raise SchemaError("plans must be immutable tuples", path=path)
        if type(self.projection) is not IR2ProjectionResult:
            raise SchemaError(
                "must be an IR2ProjectionResult",
                path=f"{path}.projection",
            )
        self.projection.validate_against(
            self.graph,
            self.fusion_plans,
            self.standalone_plans,
            f"{path}.projection",
        )
        if self.projection.state_transfers:
            raise SchemaError(
                "train forward projection cannot carry state transfers",
                path=f"{path}.projection.state_transfers",
            )

        manifest = self.graph.persistent_state_manifest
        if _is_s2_lite_train_graph(self.graph):
            _validate_s2_lite_projection_state(
                self.graph,
                self.projection,
                path,
            )
        else:
            if manifest is None or any(
                declaration.identity.kind is not StateKind.PARAMETER
                for declaration in manifest.declarations
            ):
                raise SchemaError(
                    "train forward projection requires parameter-only state",
                    path=f"{path}.graph.persistent_state_manifest",
                )
            if any(
                task.kind is SemanticTaskKind.DMA_OUT
                for dag in self.projection.dags
                for task in dag.tasks
            ):
                raise SchemaError(
                    "train forward projection cannot write persistent state",
                    path=f"{path}.projection.dags",
                )

        local_rank_by_die = {
            placement.die_id: placement.rank
            for placement in self.graph.groups[0].placements
        }
        for index, dag in enumerate(self.projection.dags):
            if dag.die_id in local_rank_by_die:
                continue
            if (
                dag.tasks
                or dag.values
                or dag.flows
                or dag.regions
                or dag.state_access_ids
                or dag.state_staging_values
                or dag.state_transfer_ids
            ):
                raise SchemaError(
                    "non-replica die DAG must be empty",
                    path=f"{path}.projection.dags[{index}]",
                )

        ce_nodes = tuple(
            node for node in self.graph.nodes if node.kind is OpKind.CE_FORWARD
        )
        if len(ce_nodes) != 1:
            raise SchemaError(
                "train forward requires exactly one CE_FORWARD node",
                path=f"{path}.graph.nodes",
            )
        ce_node = ce_nodes[0]
        value_index = {value.id: value for value in self.graph.values}
        if (
            len(ce_node.inputs) != 2
            or len(ce_node.outputs) != 1
            or value_index[ce_node.inputs[0]].dtype is not DType.FP16
            or value_index[ce_node.inputs[1]].dtype is not DType.INT32
            or value_index[ce_node.outputs[0]].dtype is not DType.FP32
        ):
            raise SchemaError(
                "CE_FORWARD requires FP16 logits, INT32 labels, and FP32 loss",
                path=f"{path}.graph.nodes",
            )
        incoming = tuple(
            edge for edge in self.graph.edges if edge.destination_node == ce_node.id
        )
        if len(incoming) != 1:
            raise SchemaError(
                "CE_FORWARD must have one logits producer dependency",
                path=f"{path}.graph.edges",
            )
        predecessor_id = incoming[0].source_node
        for die_id, rank in local_rank_by_die.items():
            dag = next(item for item in self.projection.dags if item.die_id == die_id)
            tasks = tuple(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                and task.origin_ref.op_id == ce_node.id
            )
            if len(tasks) != 1:
                raise SchemaError(
                    "each replica rank requires one CE_FORWARD task",
                    path=f"{path}.projection.dags",
                )
            task = tasks[0]
            if (
                task.id != f"task.{ce_node.id}.rank.{rank}.comp"
                or task.kind is not SemanticTaskKind.COMP
                or task.read_values != ce_node.inputs
                or task.write_values != ce_node.outputs
                or task.deps
                != (f"task.{predecessor_id}.rank.{rank}.comp",)
                or task.compute is None
                or tuple(item.role for item in task.compute.inputs)
                != ("logits", "labels")
                or tuple(item.role for item in task.compute.outputs) != ("loss",)
            ):
                raise SchemaError(
                    "CE_FORWARD task role/arity/dependency contract is not exact",
                    path=f"{path}.projection.dags",
                )
        expected_id = stable_artifact_id(
            "train_projected_replica",
            self._semantic_key(),
            schema_version=TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable replica id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainReplicaInterDiePlans,
        path: str,
    ) -> None:
        source.validate(f"{path}.source")
        self.validate(path)
        if (
            self.replica_index != source.replica_index
            or self.source_replica_plan_id != source.id
            or self.source_ir1_id != source.graph.id
            or self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
        ):
            raise SchemaError(
                "must preserve the exact replica plans and graph",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TrainProjectedIR2:
    schema_version: str
    producer_pass: str
    id: str
    source_planned_carrier_id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainProjectedReplica, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainInterDiePlannedIR1,
        context: ProjectToIR2Context,
        replicas: tuple[TrainProjectedReplica, ...],
    ) -> "TrainProjectedIR2":
        semantic_key = {
            "source_planned_carrier_id": source.id,
            "source_partitioned_carrier_id": source.source_partitioned_id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": context.id,
            "train_structure": source.train_structure,
            "dp_degree": source.dp_degree,
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
            producer_pass="train_project_to_ir2",
            id=stable_artifact_id(
                "train_projected_ir2",
                semantic_key,
                schema_version=TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_planned_carrier_id": self.source_planned_carrier_id,
            "source_partitioned_carrier_id": self.source_partitioned_carrier_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "train_structure": self.train_structure,
            "dp_degree": self.dp_degree,
            "replicas": self.replicas,
        }

    def validate(self, path: str = "train_projected_ir2") -> None:
        if self.schema_version != TRAIN_PROJECTED_IR2_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_project_to_ir2":
            raise SchemaError(
                "must be 'train_project_to_ir2'",
                path=f"{path}.producer_pass",
            )
        for name in (
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError(
                "must contain exact DP replica projection coverage",
                path=f"{path}.replicas",
            )
        if tuple(item.replica_index for item in self.replicas) != tuple(
            range(self.dp_degree)
        ):
            raise SchemaError(
                "replica projections must use canonical DP order",
                path=f"{path}.replicas",
            )
        projection_ids: set[str] = set()
        task_ids: set[str] = set()
        flow_ids: set[str] = set()
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainProjectedReplica:
                raise SchemaError("must be a TrainProjectedReplica", path=replica_path)
            replica.validate(replica_path)
            if replica.projection.id in projection_ids:
                raise SchemaError(
                    "projection ids must be replica-distinct",
                    path=f"{replica_path}.projection.id",
                )
            projection_ids.add(replica.projection.id)
            local_task_ids = {
                task.id
                for dag in replica.projection.dags
                for task in dag.tasks
            }
            if task_ids.intersection(local_task_ids):
                raise SchemaError(
                    "task ids must be replica-distinct",
                    path=f"{replica_path}.projection.dags",
                )
            task_ids.update(local_task_ids)
            local_flow_ids = {
                flow.id
                for dag in replica.projection.dags
                for flow in dag.flows
            }
            if flow_ids.intersection(local_flow_ids):
                raise SchemaError(
                    "flow ids must be replica-distinct",
                    path=f"{replica_path}.projection.dags",
                )
            flow_ids.update(local_flow_ids)
        expected_id = stable_artifact_id(
            "train_projected_ir2",
            self._semantic_key(),
            schema_version=TRAIN_PROJECTED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainInterDiePlannedIR1,
        context: ProjectToIR2Context,
        path: str = "train_projected_ir2",
    ) -> None:
        source.validate("source")
        context.validate("project_to_ir2_context")
        self.validate(path)
        if context.state_transfers:
            raise SchemaError(
                "train forward projection context must not contain state transfers",
                path="project_to_ir2_context.state_transfers",
            )
        if (
            self.source_planned_carrier_id != source.id
            or self.source_partitioned_carrier_id != source.source_partitioned_id
            or self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
            or self.planning_context_id != source.planning_context_id
            or self.projection_context_id != context.id
            or self.train_structure != source.train_structure
            or self.dp_degree != source.dp_degree
            or len(self.replicas) != len(source.replicas)
        ):
            raise SchemaError(
                "must preserve complete train planning provenance",
                path=path,
            )
        for index, (replica, source_replica) in enumerate(
            zip(self.replicas, source.replicas)
        ):
            replica.validate_against(
                source_replica,
                f"{path}.replicas[{index}]",
            )


@dataclass(frozen=True, slots=True)
class Stage4ProjectedIR2:
    """Exact single-graph Stage 4 ProjectToIR2 carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_planned_carrier_id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult

    @classmethod
    def create(
        cls,
        *,
        source: Stage4InterDiePlannedIR1,
        context: Stage4ProjectToIR2Context,
        projection: IR2ProjectionResult,
    ) -> "Stage4ProjectedIR2":
        semantic_key = {
            "source_planned_carrier_id": source.id,
            "source_partitioned_carrier_id": (
                source.source_partitioned_carrier_id
            ),
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": context.id,
            "pd_plan": source.pd_plan,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(
                plan.id for plan in source.fusion_plans
            ),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": projection.id,
        }
        return cls(
            schema_version=STAGE4_PROJECTED_IR2_SCHEMA_VERSION,
            producer_pass="project_to_ir2",
            id=stable_artifact_id(
                "stage4_projected_ir2",
                semantic_key,
                schema_version=STAGE4_PROJECTED_IR2_SCHEMA_VERSION,
            ),
            source_planned_carrier_id=source.id,
            source_partitioned_carrier_id=(
                source.source_partitioned_carrier_id
            ),
            placement_context_id=source.placement_context_id,
            partition_context_id=source.partition_context_id,
            planning_context_id=source.planning_context_id,
            projection_context_id=context.id,
            pd_plan=source.pd_plan,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=projection,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_planned_carrier_id": self.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                self.source_partitioned_carrier_id
            ),
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
        }

    def validate(self, path: str = "stage4_projected_ir2") -> None:
        if self.schema_version != STAGE4_PROJECTED_IR2_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "project_to_ir2":
            raise SchemaError(
                "must be 'project_to_ir2'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError(
                "must be a Stage4PdPlan",
                path=f"{path}.pd_plan",
            )
        self.pd_plan.validate(f"{path}.pd_plan")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError(
                "must preserve fusion_partition IR1",
                path=f"{path}.graph.producer_pass",
            )
        if self.graph.pd_plan_id != self.pd_plan.id:
            raise SchemaError(
                "graph must reference embedded PD plan",
                path=f"{path}.graph.pd_plan_id",
            )
        if type(self.fusion_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.fusion_plans",
            )
        if type(self.standalone_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.standalone_plans",
            )
        if type(self.projection) is not IR2ProjectionResult:
            raise SchemaError(
                "must be an IR2ProjectionResult",
                path=f"{path}.projection",
            )
        if self.projection.producer_pass != "project_to_ir2":
            raise SchemaError(
                "must be produced by project_to_ir2",
                path=f"{path}.projection.producer_pass",
            )
        self.projection.validate_against(
            self.graph,
            self.fusion_plans,
            self.standalone_plans,
            f"{path}.projection",
        )
        if self.pd_plan.mode is Stage4PdMode.FUSED and (
            self.pd_plan.handoffs
            or self.graph.cross_routes
            or self.projection.state_transfers
        ):
            raise SchemaError(
                "fused PD must not contain handoffs, cross routes, or transfers",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_projected_ir2",
            self._semantic_key(),
            schema_version=STAGE4_PROJECTED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: Stage4InterDiePlannedIR1,
        context: Stage4ProjectToIR2Context,
        path: str = "stage4_projected_ir2",
    ) -> None:
        if type(source) is not Stage4InterDiePlannedIR1:
            raise SchemaError(
                "must be a Stage4InterDiePlannedIR1",
                path="source",
            )
        if type(context) is not Stage4ProjectToIR2Context:
            raise SchemaError(
                "must be a Stage4ProjectToIR2Context",
                path="stage4_project_to_ir2_context",
            )
        source.validate("source")
        context.validate("stage4_project_to_ir2_context")
        self.validate(path)
        if self.source_planned_carrier_id != source.id:
            raise SchemaError(
                "must match source planned carrier",
                path=f"{path}.source_planned_carrier_id",
            )
        if (
            self.source_partitioned_carrier_id
            != source.source_partitioned_carrier_id
        ):
            raise SchemaError(
                "must preserve source partitioned carrier",
                path=f"{path}.source_partitioned_carrier_id",
            )
        for field_name in (
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream context provenance",
                    path=f"{path}.{field_name}",
                )
        if self.projection_context_id != context.id:
            raise SchemaError(
                "must match projection context",
                path=f"{path}.projection_context_id",
            )
        if (
            self.pd_plan != source.pd_plan
            or self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
        ):
            raise SchemaError(
                "must preserve the complete planned payload",
                path=path,
            )
        from ..passes.project_to_ir2 import (
            build_stage4_project_state_transfers,
        )

        expected_transfers = build_stage4_project_state_transfers(source)
        if self.projection.state_transfers != expected_transfers:
            raise SchemaError(
                "projection must equal the derived Stage 4 transfers",
                path=f"{path}.projection.state_transfers",
            )


@dataclass(frozen=True, slots=True)
class TrainScheduledReplica:
    id: str
    replica_index: int
    source_projected_replica_id: str
    projected: TrainProjectedReplica
    schedule_set: IntraDieScheduleSet

    @classmethod
    def create(
        cls,
        *,
        source: TrainProjectedReplica,
        schedule_set: IntraDieScheduleSet,
    ) -> "TrainScheduledReplica":
        semantic_key = {
            "replica_index": source.replica_index,
            "source_projected_replica_id": source.id,
            "projected_id": source.id,
            "schedule_set_id": schedule_set.id,
        }
        return cls(
            id=stable_artifact_id(
                "train_scheduled_replica",
                semantic_key,
                schema_version=TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
            ),
            replica_index=source.replica_index,
            source_projected_replica_id=source.id,
            projected=source,
            schedule_set=schedule_set,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "replica_index": self.replica_index,
            "source_projected_replica_id": self.source_projected_replica_id,
            "projected_id": self.projected.id,
            "schedule_set_id": self.schedule_set.id,
        }

    def validate(self, path: str) -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        validate_nonempty(
            self.source_projected_replica_id,
            f"{path}.source_projected_replica_id",
        )
        if type(self.projected) is not TrainProjectedReplica:
            raise SchemaError(
                "must be a TrainProjectedReplica",
                path=f"{path}.projected",
            )
        self.projected.validate(f"{path}.projected")
        if (
            self.replica_index != self.projected.replica_index
            or self.source_projected_replica_id != self.projected.id
        ):
            raise SchemaError(
                "must preserve projected replica identity",
                path=path,
            )
        if type(self.schedule_set) is not IntraDieScheduleSet:
            raise SchemaError(
                "must be an IntraDieScheduleSet",
                path=f"{path}.schedule_set",
            )
        self.schedule_set.validate_against(
            self.projected.projection,
            self.projected.graph,
            f"{path}.schedule_set",
        )

        graph = self.projected.graph
        local_rank_by_die = {
            placement.die_id: placement.rank
            for placement in graph.groups[0].placements
        }
        route_ids = {
            route.id for route in graph.groups[0].embedding.routes
        }
        ce_node = next(
            node for node in graph.nodes if node.kind is OpKind.CE_FORWARD
        )
        if type(ce_node.workload) is not CrossEntropyForwardWorkload:
            raise SchemaError(
                "CE_FORWARD requires a typed workload",
                path=f"{path}.projected.graph.nodes",
            )
        ce_workload = ce_node.workload
        incoming = tuple(
            edge for edge in graph.edges if edge.destination_node == ce_node.id
        )
        predecessor_id = incoming[0].source_node
        schedule_by_die = {
            schedule.die_id: schedule
            for schedule in self.schedule_set.schedules
        }
        dag_by_die = {
            dag.die_id: dag for dag in self.projected.projection.dags
        }
        for die_id, schedule in schedule_by_die.items():
            if die_id not in local_rank_by_die:
                if (
                    schedule.placements
                    or schedule.buffer_bindings
                    or schedule.task_buffer_uses
                    or schedule.task_state_uses
                    or schedule.flow_routes
                    or schedule.runtime_bindings
                    or schedule.core_orders
                ):
                    raise SchemaError(
                        "non-replica die schedule must be empty",
                        path=f"{path}.schedule_set.schedules",
                    )
                continue
            if any(
                route.pair_route_ref not in route_ids
                for route in schedule.flow_routes
            ):
                raise SchemaError(
                    "flow route cannot reference another replica group",
                    path=f"{path}.schedule_set.schedules",
                )
            rank = local_rank_by_die[die_id]
            dag = dag_by_die[die_id]
            ce_task = next(
                task
                for task in dag.tasks
                if isinstance(task.origin_ref, OrdinaryNodeOrigin)
                and task.origin_ref.op_id == ce_node.id
            )
            ce_backward_task = next(
                (
                    task
                    for task in dag.tasks
                    if task.op_kind is OpKind.CE_BACKWARD
                ),
                None,
            )
            predecessor_task_id = f"task.{predecessor_id}.rank.{rank}.comp"
            placements = tuple(
                placement
                for placement in schedule.placements
                if placement.task_id == ce_task.id
            )
            if len(placements) != 1:
                raise SchemaError(
                    "CE_FORWARD must have one task placement per rank",
                    path=f"{path}.schedule_set.schedules",
                )
            core_order = next(
                order
                for order in schedule.core_orders
                if order.core_id == placements[0].core_id
            )
            predecessor_position = core_order.task_ids.index(predecessor_task_id)
            ce_position = core_order.task_ids.index(ce_task.id)
            if ce_position != predecessor_position + 1:
                raise SchemaError(
                    "LM-head must immediately precede CE_FORWARD on one core",
                    path=f"{path}.schedule_set.schedules",
                )
            if _is_s2_lite_train_graph(graph):
                if ce_backward_task is None:
                    raise SchemaError(
                        "S2-Lite schedule requires one CE_BACKWARD task",
                        path=f"{path}.schedule_set.schedules",
                    )
                ce_backward_position = core_order.task_ids.index(
                    ce_backward_task.id
                )
                if ce_backward_position <= ce_position:
                    raise SchemaError(
                        "CE_BACKWARD must follow CE_FORWARD",
                        path=f"{path}.schedule_set.schedules",
                    )
                ce_input_lifetime_end = ce_backward_position + 1
            else:
                ce_input_lifetime_end = ce_position + 1

            uses = tuple(
                use for use in schedule.task_buffer_uses
                if use.task_id == ce_task.id
            )
            if tuple(
                (use.role, use.access, use.operand_index)
                for use in uses
            ) != (
                (BufferUseRole.COMP_INPUT, BufferAccess.READ, 0),
                (BufferUseRole.COMP_INPUT, BufferAccess.READ, 1),
                (BufferUseRole.COMP_OUTPUT, BufferAccess.WRITE, 0),
            ):
                raise SchemaError(
                    "CE_FORWARD task buffer roles/arity are not exact",
                    path=f"{path}.schedule_set.schedules",
                )
            binding_index = {
                binding.id: binding for binding in schedule.buffer_bindings
            }
            bindings = tuple(binding_index[use.binding_id] for use in uses)
            expected = (
                (
                    ce_node.inputs[0],
                    ce_workload.logits_dtype,
                    math.prod(ce_workload.rank_logits_shape) * 2,
                    predecessor_position,
                    ce_input_lifetime_end,
                ),
                (
                    ce_node.inputs[1],
                    ce_workload.label_dtype,
                    math.prod(ce_workload.rank_label_shape) * 4,
                    ce_position,
                    ce_input_lifetime_end,
                ),
                (
                    ce_node.outputs[0],
                    ce_workload.loss_dtype,
                    math.prod(ce_workload.rank_loss_shape) * 4,
                    ce_position,
                    ce_position + 1,
                ),
            )
            if any(
                binding.value_id != value_id
                or binding.dtype is not dtype
                or binding.size_bytes != size_bytes
                or binding.lifetime_start != lifetime_start
                or binding.lifetime_end_exclusive != lifetime_end
                for binding, (
                    value_id,
                    dtype,
                    size_bytes,
                    lifetime_start,
                    lifetime_end,
                ) in zip(bindings, expected)
            ):
                raise SchemaError(
                    "CE_FORWARD buffer dtype/size/lifetime is not exact",
                    path=f"{path}.schedule_set.schedules",
                )
        expected_id = stable_artifact_id(
            "train_scheduled_replica",
            self._semantic_key(),
            schema_version=TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable replica id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainProjectedReplica,
        path: str,
    ) -> None:
        source.validate(f"{path}.source")
        self.validate(path)
        if self.projected != source or self.source_projected_replica_id != source.id:
            raise SchemaError(
                "must preserve the exact projected replica",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TrainScheduledIR2:
    schema_version: str
    producer_pass: str
    id: str
    source_projected_carrier_id: str
    source_planned_carrier_id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    scheduling_context_id: str
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainScheduledReplica, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainProjectedIR2,
        context: IntraDieSchedulingContext,
        replicas: tuple[TrainScheduledReplica, ...],
    ) -> "TrainScheduledIR2":
        semantic_key = {
            "source_projected_carrier_id": source.id,
            "source_planned_carrier_id": source.source_planned_carrier_id,
            "source_partitioned_carrier_id": source.source_partitioned_carrier_id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": context.id,
            "train_structure": source.train_structure,
            "dp_degree": source.dp_degree,
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
            producer_pass="train_intra_die_schedule",
            id=stable_artifact_id(
                "train_scheduled_ir2",
                semantic_key,
                schema_version=TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projected_carrier_id": self.source_projected_carrier_id,
            "source_planned_carrier_id": self.source_planned_carrier_id,
            "source_partitioned_carrier_id": self.source_partitioned_carrier_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "train_structure": self.train_structure,
            "dp_degree": self.dp_degree,
            "replicas": self.replicas,
        }

    def validate(self, path: str = "train_scheduled_ir2") -> None:
        if self.schema_version != TRAIN_SCHEDULED_IR2_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_intra_die_schedule":
            raise SchemaError(
                "must be 'train_intra_die_schedule'",
                path=f"{path}.producer_pass",
            )
        for name in (
            "source_projected_carrier_id",
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError(
                "must contain exact DP replica schedule coverage",
                path=f"{path}.replicas",
            )
        if tuple(item.replica_index for item in self.replicas) != tuple(
            range(self.dp_degree)
        ):
            raise SchemaError(
                "replica schedules must use canonical DP order",
                path=f"{path}.replicas",
            )
        schedule_ids: set[str] = set()
        core_ids: set[int] = set()
        binding_ids: set[str] = set()
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainScheduledReplica:
                raise SchemaError("must be a TrainScheduledReplica", path=replica_path)
            replica.validate(replica_path)
            if replica.schedule_set.id in schedule_ids:
                raise SchemaError(
                    "schedule-set ids must be replica-distinct",
                    path=f"{replica_path}.schedule_set.id",
                )
            schedule_ids.add(replica.schedule_set.id)
            local_core_ids = {
                placement.core_id
                for schedule in replica.schedule_set.schedules
                for placement in schedule.placements
            }
            if core_ids.intersection(local_core_ids):
                raise SchemaError(
                    "physical core ids must be replica-disjoint",
                    path=f"{replica_path}.schedule_set.schedules",
                )
            core_ids.update(local_core_ids)
            local_binding_ids = {
                binding.id
                for schedule in replica.schedule_set.schedules
                for binding in schedule.buffer_bindings
            }
            if binding_ids.intersection(local_binding_ids):
                raise SchemaError(
                    "buffer binding ids must be replica-distinct",
                    path=f"{replica_path}.schedule_set.schedules",
                )
            binding_ids.update(local_binding_ids)
        expected_id = stable_artifact_id(
            "train_scheduled_ir2",
            self._semantic_key(),
            schema_version=TRAIN_SCHEDULED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainProjectedIR2,
        context: IntraDieSchedulingContext,
        path: str = "train_scheduled_ir2",
    ) -> None:
        source.validate("source")
        context.validate("intra_die_scheduling_context")
        self.validate(path)
        if (
            self.source_projected_carrier_id != source.id
            or self.source_planned_carrier_id != source.source_planned_carrier_id
            or self.source_partitioned_carrier_id
            != source.source_partitioned_carrier_id
            or self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
            or self.planning_context_id != source.planning_context_id
            or self.projection_context_id != source.projection_context_id
            or self.scheduling_context_id != context.id
            or self.train_structure != source.train_structure
            or self.dp_degree != source.dp_degree
            or len(self.replicas) != len(source.replicas)
        ):
            raise SchemaError(
                "must preserve complete train projection provenance",
                path=path,
            )
        for index, (replica, source_replica) in enumerate(
            zip(self.replicas, source.replicas)
        ):
            replica.validate_against(
                source_replica,
                f"{path}.replicas[{index}]",
            )


@dataclass(frozen=True, slots=True)
class Stage4ScheduledIR2:
    """Exact single-graph Stage 4 intra-die scheduling carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_projected_carrier_id: str
    source_planned_carrier_id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    scheduling_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet

    @classmethod
    def create(
        cls,
        *,
        source: Stage4ProjectedIR2,
        context: IntraDieSchedulingContext,
        schedule_set: IntraDieScheduleSet,
    ) -> "Stage4ScheduledIR2":
        semantic_key = {
            "source_projected_carrier_id": source.id,
            "source_planned_carrier_id": source.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                source.source_partitioned_carrier_id
            ),
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": context.id,
            "pd_plan": source.pd_plan,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(
                plan.id for plan in source.fusion_plans
            ),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": source.projection.id,
            "schedule_set_id": schedule_set.id,
        }
        return cls(
            schema_version=STAGE4_SCHEDULED_IR2_SCHEMA_VERSION,
            producer_pass="intra_die_schedule",
            id=stable_artifact_id(
                "stage4_scheduled_ir2",
                semantic_key,
                schema_version=STAGE4_SCHEDULED_IR2_SCHEMA_VERSION,
            ),
            source_projected_carrier_id=source.id,
            source_planned_carrier_id=source.source_planned_carrier_id,
            source_partitioned_carrier_id=(
                source.source_partitioned_carrier_id
            ),
            placement_context_id=source.placement_context_id,
            partition_context_id=source.partition_context_id,
            planning_context_id=source.planning_context_id,
            projection_context_id=source.projection_context_id,
            scheduling_context_id=context.id,
            pd_plan=source.pd_plan,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=source.projection,
            schedule_set=schedule_set,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projected_carrier_id": self.source_projected_carrier_id,
            "source_planned_carrier_id": self.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                self.source_partitioned_carrier_id
            ),
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
            "schedule_set_id": self.schedule_set.id,
        }

    def validate(self, path: str = "stage4_scheduled_ir2") -> None:
        if self.schema_version != STAGE4_SCHEDULED_IR2_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "intra_die_schedule":
            raise SchemaError(
                "must be 'intra_die_schedule'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_projected_carrier_id",
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError(
                "must be a Stage4PdPlan",
                path=f"{path}.pd_plan",
            )
        self.pd_plan.validate(f"{path}.pd_plan")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError(
                "must preserve fusion_partition IR1",
                path=f"{path}.graph.producer_pass",
            )
        if self.graph.pd_plan_id != self.pd_plan.id:
            raise SchemaError(
                "graph must reference embedded PD plan",
                path=f"{path}.graph.pd_plan_id",
            )
        if type(self.fusion_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.fusion_plans",
            )
        if type(self.standalone_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.standalone_plans",
            )
        if type(self.projection) is not IR2ProjectionResult:
            raise SchemaError(
                "must be an IR2ProjectionResult",
                path=f"{path}.projection",
            )
        if self.projection.producer_pass != "project_to_ir2":
            raise SchemaError(
                "must be produced by project_to_ir2",
                path=f"{path}.projection.producer_pass",
            )
        self.projection.validate_against(
            self.graph,
            self.fusion_plans,
            self.standalone_plans,
            f"{path}.projection",
        )
        if type(self.schedule_set) is not IntraDieScheduleSet:
            raise SchemaError(
                "must be an IntraDieScheduleSet",
                path=f"{path}.schedule_set",
            )
        if self.schedule_set.producer_pass != "intra_die_schedule":
            raise SchemaError(
                "must be produced by intra_die_schedule",
                path=f"{path}.schedule_set.producer_pass",
            )
        for index, schedule in enumerate(self.schedule_set.schedules):
            if schedule.producer_pass != "intra_die_schedule":
                raise SchemaError(
                    "must be produced by intra_die_schedule",
                    path=(
                        f"{path}.schedule_set.schedules[{index}]"
                        ".producer_pass"
                    ),
                )
        self.schedule_set.validate_against(
            self.projection,
            self.graph,
            f"{path}.schedule_set",
        )
        if self.pd_plan.mode is Stage4PdMode.FUSED and (
            self.pd_plan.handoffs
            or self.graph.cross_routes
            or self.projection.state_transfers
        ):
            raise SchemaError(
                "fused PD must not contain handoffs, cross routes, or transfers",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_scheduled_ir2",
            self._semantic_key(),
            schema_version=STAGE4_SCHEDULED_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: Stage4ProjectedIR2,
        context: IntraDieSchedulingContext,
        path: str = "stage4_scheduled_ir2",
    ) -> None:
        if type(source) is not Stage4ProjectedIR2:
            raise SchemaError(
                "must be a Stage4ProjectedIR2",
                path="source",
            )
        if type(context) is not IntraDieSchedulingContext:
            raise SchemaError(
                "must be an IntraDieSchedulingContext",
                path="intra_die_scheduling_context",
            )
        source.validate("source")
        context.validate("intra_die_scheduling_context")
        self.validate(path)
        if self.source_projected_carrier_id != source.id:
            raise SchemaError(
                "must match source projected carrier",
                path=f"{path}.source_projected_carrier_id",
            )
        for field_name in (
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if self.scheduling_context_id != context.id:
            raise SchemaError(
                "must match scheduling context",
                path=f"{path}.scheduling_context_id",
            )
        if (
            self.pd_plan != source.pd_plan
            or self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
            or self.projection != source.projection
        ):
            raise SchemaError(
                "must preserve the complete projected payload",
                path=path,
            )


def _validate_scheduled_payload(
    *,
    graph: IR1,
    fusion_plans: tuple[FusionPlan, ...],
    standalone_plans: tuple[StandaloneCollectivePlan, ...],
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
    profile_id: str,
    path: str,
) -> None:
    _validate_projected_payload(
        graph=graph,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
        projection=projection,
        profile_id=profile_id,
        path=path,
    )
    if type(schedule_set) is not IntraDieScheduleSet:
        raise SchemaError(
            "must be an IntraDieScheduleSet",
            path=f"{path}.schedule_set",
        )
    if schedule_set.producer_pass != "intra_die_schedule":
        raise SchemaError(
            "must be produced by intra_die_schedule",
            path=f"{path}.schedule_set.producer_pass",
        )
    for index, schedule in enumerate(schedule_set.schedules):
        if schedule.producer_pass != "intra_die_schedule":
            raise SchemaError(
                "must be produced by intra_die_schedule",
                path=(
                    f"{path}.schedule_set.schedules[{index}].producer_pass"
                ),
            )
    if tuple(schedule.dag_id for schedule in schedule_set.schedules) != tuple(
        dag.id for dag in projection.dags
    ):
        raise SchemaError(
            "must exactly preserve projection DAG order",
            path=f"{path}.schedule_set.schedules",
        )
    schedule_set.validate_against(
        projection,
        graph,
        f"{path}.schedule_set",
    )


@dataclass(frozen=True, slots=True)
class ScheduledProfileIR2:
    id: str
    source_projected_entry_id: str
    source_planned_entry_id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    projection_context_id: str
    scheduling_context_id: str
    profile_id: str
    weight: float
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet

    @classmethod
    def create(
        cls,
        *,
        source: ProjectedProfileIR2,
        context: IntraDieSchedulingContext,
        schedule_set: IntraDieScheduleSet,
    ) -> "ScheduledProfileIR2":
        semantic_key = {
            "source_projected_entry_id": source.id,
            "source_planned_entry_id": source.source_planned_entry_id,
            "source_partitioned_entry_id": source.source_partitioned_entry_id,
            "source_ir1_id": source.source_ir1_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": context.id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in source.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": source.projection.id,
            "schedule_set_id": schedule_set.id,
        }
        return cls(
            id=stable_artifact_id(
                "scheduled_profile_ir2",
                semantic_key,
                schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
            source_projected_entry_id=source.id,
            source_planned_entry_id=source.source_planned_entry_id,
            source_partitioned_entry_id=source.source_partitioned_entry_id,
            source_ir1_id=source.source_ir1_id,
            projection_context_id=source.projection_context_id,
            scheduling_context_id=context.id,
            profile_id=source.profile_id,
            weight=source.weight,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=source.projection,
            schedule_set=schedule_set,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projected_entry_id": self.source_projected_entry_id,
            "source_planned_entry_id": self.source_planned_entry_id,
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
            "schedule_set_id": self.schedule_set.id,
        }

    def validate(self, path: str = "scheduled_profile_ir2") -> None:
        for field_name in (
            "source_projected_entry_id",
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
            "scheduling_context_id",
            "profile_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_weight(self.weight, f"{path}.weight")
        _validate_scheduled_payload(
            graph=self.graph,
            fusion_plans=self.fusion_plans,
            standalone_plans=self.standalone_plans,
            projection=self.projection,
            schedule_set=self.schedule_set,
            profile_id=self.profile_id,
            path=path,
        )
        if self.source_ir1_id != self.graph.id:
            raise SchemaError(
                "must equal embedded graph id",
                path=f"{path}.source_ir1_id",
            )
        expected_id = stable_artifact_id(
            "scheduled_profile_ir2",
            self._semantic_key(),
            schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: ProjectedProfileIR2,
        context: IntraDieSchedulingContext,
        path: str = "scheduled_profile_ir2",
    ) -> None:
        source.validate("projected_profile_ir2")
        context.validate("intra_die_scheduling_context")
        self.validate(path)
        if self.source_projected_entry_id != source.id:
            raise SchemaError(
                "must match source projected entry",
                path=f"{path}.source_projected_entry_id",
            )
        for field_name in (
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if self.scheduling_context_id != context.id:
            raise SchemaError(
                "must match scheduling context",
                path=f"{path}.scheduling_context_id",
            )
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError("must preserve source profile and weight", path=path)
        if (
            self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
            or self.projection != source.projection
        ):
            raise SchemaError(
                "must preserve the complete projected payload",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class ScheduledIR2Bundle:
    schema_version: str
    producer_pass: str
    id: str
    source_projected_bundle_id: str
    source_inter_die_bundle_id: str
    source_partitioned_bundle_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    scheduling_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[ScheduledProfileIR2, ...]

    @classmethod
    def create(
        cls,
        *,
        source: ProjectedIR2Bundle,
        context: IntraDieSchedulingContext,
        entries: tuple[ScheduledProfileIR2, ...],
    ) -> "ScheduledIR2Bundle":
        semantic_key = {
            "source_projected_bundle_id": source.id,
            "source_inter_die_bundle_id": source.source_inter_die_bundle_id,
            "source_partitioned_bundle_id": source.source_partitioned_bundle_id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": context.id,
            "source_profiles": source.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
            producer_pass="intra_die_schedule",
            id=stable_artifact_id(
                "scheduled_ir2_bundle",
                semantic_key,
                schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projected_bundle_id": self.source_projected_bundle_id,
            "source_inter_die_bundle_id": self.source_inter_die_bundle_id,
            "source_partitioned_bundle_id": self.source_partitioned_bundle_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "scheduled_ir2_bundle") -> None:
        if self.schema_version != SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "intra_die_schedule":
            raise SchemaError(
                "must be 'intra_die_schedule'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_projected_bundle_id",
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_profile_manifest(
            self.source_profiles,
            path=f"{path}.source_profiles",
        )
        if type(self.entries) is not tuple or len(self.entries) != len(
            self.source_profiles
        ):
            raise SchemaError(
                "must contain one immutable entry per profile",
                path=f"{path}.entries",
            )
        source_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not ScheduledProfileIR2:
                raise SchemaError(
                    "must be a ScheduledProfileIR2",
                    path=entry_path,
                )
            entry.validate(entry_path)
            if (
                entry.profile_id != profile.profile_id
                or entry.graph.profile != profile.key
            ):
                raise SchemaError(
                    "must match source profile order",
                    path=f"{entry_path}.profile_id",
                )
            if entry.weight != profile.weight:
                raise SchemaError(
                    "must match source profile weight",
                    path=f"{entry_path}.weight",
                )
            if entry.projection_context_id != self.projection_context_id:
                raise SchemaError(
                    "must match bundle projection context",
                    path=f"{entry_path}.projection_context_id",
                )
            if entry.scheduling_context_id != self.scheduling_context_id:
                raise SchemaError(
                    "must match bundle scheduling context",
                    path=f"{entry_path}.scheduling_context_id",
                )
            if entry.source_projected_entry_id in source_ids:
                raise SchemaError(
                    "duplicate source projected entry",
                    path=f"{entry_path}.source_projected_entry_id",
                )
            source_ids.add(entry.source_projected_entry_id)
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
                    "all profiles must share fabric, groups, and fusion partition",
                    path=f"{entry_path}.graph",
                )
        expected_id = stable_artifact_id(
            "scheduled_ir2_bundle",
            self._semantic_key(),
            schema_version=SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: ProjectedIR2Bundle,
        context: IntraDieSchedulingContext,
        path: str = "scheduled_ir2_bundle",
    ) -> None:
        source.validate("projected_ir2_bundle")
        context.validate("intra_die_scheduling_context")
        self.validate(path)
        if self.source_projected_bundle_id != source.id:
            raise SchemaError(
                "must match source projected bundle",
                path=f"{path}.source_projected_bundle_id",
            )
        for field_name in (
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if self.scheduling_context_id != context.id:
            raise SchemaError(
                "must match scheduling context",
                path=f"{path}.scheduling_context_id",
            )
        if (
            self.source_profiles != source.source_profiles
            or len(self.entries) != len(source.entries)
        ):
            raise SchemaError(
                "must preserve complete profile manifest",
                path=f"{path}.source_profiles",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry.validate_against(
                source_entry,
                context,
                f"{path}.entries[{index}]",
            )


def _validate_global_payload(
    *,
    graph: IR1,
    fusion_plans: tuple[FusionPlan, ...],
    standalone_plans: tuple[StandaloneCollectivePlan, ...],
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
    global_dag: GlobalActionDAG,
    profile_id: str,
    path: str,
) -> None:
    _validate_scheduled_payload(
        graph=graph,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
        projection=projection,
        schedule_set=schedule_set,
        profile_id=profile_id,
        path=path,
    )
    if type(global_dag) is not GlobalActionDAG:
        raise SchemaError(
            "must be a GlobalActionDAG",
            path=f"{path}.global_dag",
        )
    if global_dag.producer_pass != "global_action_dag":
        raise SchemaError(
            "must be produced by global_action_dag",
            path=f"{path}.global_dag.producer_pass",
        )
    global_dag.validate_against(
        graph,
        projection,
        schedule_set,
        f"{path}.global_dag",
    )


@dataclass(frozen=True, slots=True)
class Stage4GlobalAction:
    """Exact single-graph Stage 4 GlobalAction carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_scheduled_carrier_id: str
    source_projected_carrier_id: str
    source_planned_carrier_id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    scheduling_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG

    @classmethod
    def create(
        cls,
        *,
        source: Stage4ScheduledIR2,
        global_dag: GlobalActionDAG,
    ) -> "Stage4GlobalAction":
        semantic_key = {
            "source_scheduled_carrier_id": source.id,
            "source_projected_carrier_id": (
                source.source_projected_carrier_id
            ),
            "source_planned_carrier_id": source.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                source.source_partitioned_carrier_id
            ),
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "pd_plan": source.pd_plan,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(
                plan.id for plan in source.fusion_plans
            ),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": source.projection.id,
            "schedule_set_id": source.schedule_set.id,
            "global_dag_id": global_dag.id,
        }
        return cls(
            schema_version=STAGE4_GLOBAL_ACTION_SCHEMA_VERSION,
            producer_pass="global_action_dag",
            id=stable_artifact_id(
                "stage4_global_action",
                semantic_key,
                schema_version=STAGE4_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            source_scheduled_carrier_id=source.id,
            source_projected_carrier_id=(
                source.source_projected_carrier_id
            ),
            source_planned_carrier_id=source.source_planned_carrier_id,
            source_partitioned_carrier_id=(
                source.source_partitioned_carrier_id
            ),
            placement_context_id=source.placement_context_id,
            partition_context_id=source.partition_context_id,
            planning_context_id=source.planning_context_id,
            projection_context_id=source.projection_context_id,
            scheduling_context_id=source.scheduling_context_id,
            pd_plan=source.pd_plan,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=source.projection,
            schedule_set=source.schedule_set,
            global_dag=global_dag,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_scheduled_carrier_id": self.source_scheduled_carrier_id,
            "source_projected_carrier_id": self.source_projected_carrier_id,
            "source_planned_carrier_id": self.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                self.source_partitioned_carrier_id
            ),
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
            "schedule_set_id": self.schedule_set.id,
            "global_dag_id": self.global_dag.id,
        }

    def validate(self, path: str = "stage4_global_action") -> None:
        if self.schema_version != STAGE4_GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "global_action_dag":
            raise SchemaError(
                "must be 'global_action_dag'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_scheduled_carrier_id",
            "source_projected_carrier_id",
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError(
                "must be a Stage4PdPlan",
                path=f"{path}.pd_plan",
            )
        self.pd_plan.validate(f"{path}.pd_plan")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if (
            self.graph.producer_pass != "fusion_partition"
            or self.graph.pd_plan_id != self.pd_plan.id
        ):
            raise SchemaError(
                "must preserve the planned Stage 4 graph",
                path=f"{path}.graph",
            )
        if type(self.fusion_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.fusion_plans",
            )
        if type(self.standalone_plans) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.standalone_plans",
            )
        if type(self.projection) is not IR2ProjectionResult:
            raise SchemaError(
                "must be an IR2ProjectionResult",
                path=f"{path}.projection",
            )
        if self.projection.producer_pass != "project_to_ir2":
            raise SchemaError(
                "must be produced by project_to_ir2",
                path=f"{path}.projection.producer_pass",
            )
        self.projection.validate_against(
            self.graph,
            self.fusion_plans,
            self.standalone_plans,
            f"{path}.projection",
        )
        if type(self.schedule_set) is not IntraDieScheduleSet:
            raise SchemaError(
                "must be an IntraDieScheduleSet",
                path=f"{path}.schedule_set",
            )
        if self.schedule_set.producer_pass != "intra_die_schedule":
            raise SchemaError(
                "must be produced by intra_die_schedule",
                path=f"{path}.schedule_set.producer_pass",
            )
        for index, schedule in enumerate(self.schedule_set.schedules):
            if schedule.producer_pass != "intra_die_schedule":
                raise SchemaError(
                    "must be produced by intra_die_schedule",
                    path=(
                        f"{path}.schedule_set.schedules[{index}]"
                        ".producer_pass"
                    ),
                )
        self.schedule_set.validate_against(
            self.projection,
            self.graph,
            f"{path}.schedule_set",
        )
        if type(self.global_dag) is not GlobalActionDAG:
            raise SchemaError(
                "must be a GlobalActionDAG",
                path=f"{path}.global_dag",
            )
        if self.global_dag.producer_pass != "global_action_dag":
            raise SchemaError(
                "must be produced by global_action_dag",
                path=f"{path}.global_dag.producer_pass",
            )
        self.global_dag.validate_against(
            self.graph,
            self.projection,
            self.schedule_set,
            f"{path}.global_dag",
        )
        if self.pd_plan.mode is Stage4PdMode.FUSED and (
            self.pd_plan.handoffs
            or self.graph.cross_routes
            or self.projection.state_transfers
        ):
            raise SchemaError(
                "fused PD must not contain handoffs, cross routes, or transfers",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_global_action",
            self._semantic_key(),
            schema_version=STAGE4_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: Stage4ScheduledIR2,
        path: str = "stage4_global_action",
    ) -> None:
        if type(source) is not Stage4ScheduledIR2:
            raise SchemaError(
                "must be a Stage4ScheduledIR2",
                path="source",
            )
        source.validate("source")
        self.validate(path)
        if self.source_scheduled_carrier_id != source.id:
            raise SchemaError(
                "must match source scheduled carrier",
                path=f"{path}.source_scheduled_carrier_id",
            )
        for field_name in (
            "source_projected_carrier_id",
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if (
            self.pd_plan != source.pd_plan
            or self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
            or self.projection != source.projection
            or self.schedule_set != source.schedule_set
        ):
            raise SchemaError(
                "must preserve the complete scheduled payload",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class GlobalActionProfile:
    id: str
    source_scheduled_entry_id: str
    source_projected_entry_id: str
    source_planned_entry_id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    projection_context_id: str
    scheduling_context_id: str
    profile_id: str
    weight: float
    graph: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG

    @classmethod
    def create(
        cls,
        *,
        source: ScheduledProfileIR2,
        global_dag: GlobalActionDAG,
    ) -> "GlobalActionProfile":
        semantic_key = {
            "source_scheduled_entry_id": source.id,
            "source_projected_entry_id": source.source_projected_entry_id,
            "source_planned_entry_id": source.source_planned_entry_id,
            "source_partitioned_entry_id": source.source_partitioned_entry_id,
            "source_ir1_id": source.source_ir1_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "graph_id": source.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in source.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in source.standalone_plans
            ),
            "projection_id": source.projection.id,
            "schedule_set_id": source.schedule_set.id,
            "global_dag_id": global_dag.id,
        }
        return cls(
            id=stable_artifact_id(
                "global_action_profile",
                semantic_key,
                schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
            ),
            source_scheduled_entry_id=source.id,
            source_projected_entry_id=source.source_projected_entry_id,
            source_planned_entry_id=source.source_planned_entry_id,
            source_partitioned_entry_id=source.source_partitioned_entry_id,
            source_ir1_id=source.source_ir1_id,
            projection_context_id=source.projection_context_id,
            scheduling_context_id=source.scheduling_context_id,
            profile_id=source.profile_id,
            weight=source.weight,
            graph=source.graph,
            fusion_plans=source.fusion_plans,
            standalone_plans=source.standalone_plans,
            projection=source.projection,
            schedule_set=source.schedule_set,
            global_dag=global_dag,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_scheduled_entry_id": self.source_scheduled_entry_id,
            "source_projected_entry_id": self.source_projected_entry_id,
            "source_planned_entry_id": self.source_planned_entry_id,
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
            "fusion_plan_ids": tuple(plan.id for plan in self.fusion_plans),
            "standalone_plan_ids": tuple(
                plan.id for plan in self.standalone_plans
            ),
            "projection_id": self.projection.id,
            "schedule_set_id": self.schedule_set.id,
            "global_dag_id": self.global_dag.id,
        }

    def validate(self, path: str = "global_action_profile") -> None:
        for field_name in (
            "source_scheduled_entry_id",
            "source_projected_entry_id",
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
            "scheduling_context_id",
            "profile_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_weight(self.weight, f"{path}.weight")
        _validate_global_payload(
            graph=self.graph,
            fusion_plans=self.fusion_plans,
            standalone_plans=self.standalone_plans,
            projection=self.projection,
            schedule_set=self.schedule_set,
            global_dag=self.global_dag,
            profile_id=self.profile_id,
            path=path,
        )
        if self.source_ir1_id != self.graph.id:
            raise SchemaError(
                "must equal embedded graph id",
                path=f"{path}.source_ir1_id",
            )
        expected_id = stable_artifact_id(
            "global_action_profile",
            self._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: ScheduledProfileIR2,
        path: str = "global_action_profile",
    ) -> None:
        source.validate("scheduled_profile_ir2")
        self.validate(path)
        if self.source_scheduled_entry_id != source.id:
            raise SchemaError(
                "must match source scheduled entry",
                path=f"{path}.source_scheduled_entry_id",
            )
        for field_name in (
            "source_projected_entry_id",
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError("must preserve source profile and weight", path=path)
        if (
            self.graph != source.graph
            or self.fusion_plans != source.fusion_plans
            or self.standalone_plans != source.standalone_plans
            or self.projection != source.projection
            or self.schedule_set != source.schedule_set
        ):
            raise SchemaError(
                "must preserve the complete scheduled payload",
                path=path,
            )

    def lowering_context(self) -> LoweringContext:
        from ..lowering.context import LoweringContext

        self.validate("global_action_profile")
        context = LoweringContext(
            ir1=self.graph,
            fusion_plans=self.fusion_plans,
            standalone_plans=self.standalone_plans,
            projection=self.projection,
            schedule_set=self.schedule_set,
            global_dag=self.global_dag,
        )
        context.validate("lowering_context")
        return context


@dataclass(frozen=True, slots=True)
class GlobalActionBundle:
    schema_version: str
    producer_pass: str
    id: str
    source_scheduled_bundle_id: str
    source_projected_bundle_id: str
    source_inter_die_bundle_id: str
    source_partitioned_bundle_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    projection_context_id: str
    scheduling_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[GlobalActionProfile, ...]

    @classmethod
    def create(
        cls,
        *,
        source: ScheduledIR2Bundle,
        entries: tuple[GlobalActionProfile, ...],
    ) -> "GlobalActionBundle":
        semantic_key = {
            "source_scheduled_bundle_id": source.id,
            "source_projected_bundle_id": source.source_projected_bundle_id,
            "source_inter_die_bundle_id": source.source_inter_die_bundle_id,
            "source_partitioned_bundle_id": source.source_partitioned_bundle_id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "source_profiles": source.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
            producer_pass="global_action_dag",
            id=stable_artifact_id(
                "global_action_bundle",
                semantic_key,
                schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_scheduled_bundle_id": self.source_scheduled_bundle_id,
            "source_projected_bundle_id": self.source_projected_bundle_id,
            "source_inter_die_bundle_id": self.source_inter_die_bundle_id,
            "source_partitioned_bundle_id": self.source_partitioned_bundle_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "global_action_bundle") -> None:
        if self.schema_version != GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "global_action_dag":
            raise SchemaError(
                "must be 'global_action_dag'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_scheduled_bundle_id",
            "source_projected_bundle_id",
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_profile_manifest(
            self.source_profiles,
            path=f"{path}.source_profiles",
        )
        if type(self.entries) is not tuple or len(self.entries) != len(
            self.source_profiles
        ):
            raise SchemaError(
                "must contain one immutable entry per profile",
                path=f"{path}.entries",
            )
        source_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not GlobalActionProfile:
                raise SchemaError(
                    "must be a GlobalActionProfile",
                    path=entry_path,
                )
            entry.validate(entry_path)
            if (
                entry.profile_id != profile.profile_id
                or entry.graph.profile != profile.key
            ):
                raise SchemaError(
                    "must match source profile order",
                    path=f"{entry_path}.profile_id",
                )
            if entry.weight != profile.weight:
                raise SchemaError(
                    "must match source profile weight",
                    path=f"{entry_path}.weight",
                )
            if entry.projection_context_id != self.projection_context_id:
                raise SchemaError(
                    "must match bundle projection context",
                    path=f"{entry_path}.projection_context_id",
                )
            if entry.scheduling_context_id != self.scheduling_context_id:
                raise SchemaError(
                    "must match bundle scheduling context",
                    path=f"{entry_path}.scheduling_context_id",
                )
            if entry.source_scheduled_entry_id in source_ids:
                raise SchemaError(
                    "duplicate source scheduled entry",
                    path=f"{entry_path}.source_scheduled_entry_id",
                )
            source_ids.add(entry.source_scheduled_entry_id)
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
                    "all profiles must share fabric, groups, and fusion partition",
                    path=f"{entry_path}.graph",
                )
        expected_id = stable_artifact_id(
            "global_action_bundle",
            self._semantic_key(),
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: ScheduledIR2Bundle,
        path: str = "global_action_bundle",
    ) -> None:
        source.validate("scheduled_ir2_bundle")
        self.validate(path)
        if self.source_scheduled_bundle_id != source.id:
            raise SchemaError(
                "must match source scheduled bundle",
                path=f"{path}.source_scheduled_bundle_id",
            )
        for field_name in (
            "source_projected_bundle_id",
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream provenance",
                    path=f"{path}.{field_name}",
                )
        if (
            self.source_profiles != source.source_profiles
            or len(self.entries) != len(source.entries)
        ):
            raise SchemaError(
                "must preserve complete profile manifest",
                path=f"{path}.source_profiles",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry.validate_against(
                source_entry,
                f"{path}.entries[{index}]",
            )


__all__ = [
    "PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION",
    "STAGE4_PROJECT_TO_IR2_CONTEXT_SCHEMA_VERSION",
    "INTRADIE_SCHEDULING_CONTEXT_SCHEMA_VERSION",
    "PROJECTED_IR2_BUNDLE_SCHEMA_VERSION",
    "STAGE4_PROJECTED_IR2_SCHEMA_VERSION",
    "TRAIN_PROJECTED_IR2_SCHEMA_VERSION",
    "STAGE4_SCHEDULED_IR2_SCHEMA_VERSION",
    "TRAIN_SCHEDULED_IR2_SCHEMA_VERSION",
    "SCHEDULED_IR2_BUNDLE_SCHEMA_VERSION",
    "GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION",
    "STAGE4_GLOBAL_ACTION_SCHEMA_VERSION",
    "ProjectToIR2Contract",
    "IntraDieSchedulingContract",
    "ProjectToIR2Context",
    "Stage4ProjectToIR2Context",
    "IntraDieSchedulingContext",
    "ProjectedProfileIR2",
    "ProjectedIR2Bundle",
    "Stage4ProjectedIR2",
    "TrainProjectedIR2",
    "TrainProjectedReplica",
    "Stage4ScheduledIR2",
    "TrainScheduledIR2",
    "TrainScheduledReplica",
    "ScheduledProfileIR2",
    "ScheduledIR2Bundle",
    "GlobalActionProfile",
    "GlobalActionBundle",
    "Stage4GlobalAction",
]
