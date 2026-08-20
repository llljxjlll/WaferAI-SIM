"""Versioned, profile-complete N6 lowering and linking provenance wrappers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from .artifact_manifest import (
    CommandFragment,
    FragmentKind,
    LinkedProgramManifest,
    RecordOpcode,
    RegionManifest,
)
from .common import stable_artifact_id, validate_nonempty
from .ir2 import SemanticTaskKind
from .logical import ProfileEntry, _validate_profile_manifest
from .n5 import (
    GlobalActionBundle,
    GlobalActionProfile,
    Stage4GlobalAction,
)
from .stage4_pd import Stage4PdPlan


LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.lowered_program_bundle/v1alpha9"
)
LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.linked_program_bundle/v1alpha10"
)
STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.stage4_lowered_program/v1alpha5"
)
STAGE4_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.stage4_linked_program/v1alpha5"
)


def _validate_weight(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SchemaError("must be a finite positive float", path=path)


def _leaf_fragment(
    fragment: CommandFragment | RegionManifest,
) -> CommandFragment:
    if type(fragment) is RegionManifest:
        return fragment.fragment
    if type(fragment) is CommandFragment:
        return fragment
    raise SchemaError(
        "must be a CommandFragment or RegionManifest",
        path="fragments",
    )


def _leaf_fragments(
    fragments: tuple[CommandFragment | RegionManifest, ...],
) -> tuple[CommandFragment, ...]:
    return tuple(
        sorted(
            (_leaf_fragment(fragment) for fragment in fragments),
            key=lambda fragment: fragment.id,
        )
    )


def _fragments_by_leaf_id(
    fragments: tuple[CommandFragment | RegionManifest, ...],
) -> dict[str, CommandFragment | RegionManifest]:
    result: dict[str, CommandFragment | RegionManifest] = {}
    for linked in fragments:
        leaf_id = _leaf_fragment(linked).id
        if leaf_id in result:
            raise SchemaError(
                "fragments must contain unique leaf identities",
                path="fragments",
            )
        result[leaf_id] = linked
    return result


def _requires_lifecycle_records(
    fragment: CommandFragment,
    context: LoweringContext,
) -> bool:
    actions = {action.id: action for action in context.global_dag.actions}
    abi_by_binding = {
        (abi.schedule_id, abi.binding_id): abi for abi in fragment.buffer_abi
    }
    for action_id in fragment.claimed_action_ids:
        action = actions[action_id]
        assert action.core_order_index is not None
        for use in action.buffer_uses:
            abi = abi_by_binding.get(
                (action.source.schedule_id, use.binding_id)
            )
            if abi is not None and (
                abi.lifetime_start == action.core_order_index
                or abi.lifetime_end_exclusive
                == action.core_order_index + 1
            ):
                return True
    return False


def _has_lifecycle_records(fragment: CommandFragment) -> bool:
    return any(
        record.opcode in (
            RecordOpcode.SRAM_ALLOC_AT,
            RecordOpcode.SRAM_FREE,
        )
        for stream in fragment.core_streams
        for record in stream.records
    )


def _validate_lowered_fragments(
    fragments: tuple[CommandFragment | RegionManifest, ...],
    context: LoweringContext,
    path: str,
) -> None:
    if type(fragments) is not tuple or not fragments:
        raise SchemaError(
            "must contain an immutable non-empty fragment tuple",
            path=path,
        )
    leaf_ids: list[str] = []
    claimed_action_ids: list[str] = []
    for index, fragment in enumerate(fragments):
        fragment_path = f"{path}[{index}]"
        if type(fragment) is CommandFragment:
            if fragment.kind is FragmentKind.ISA_REGION:
                raise SchemaError(
                    "ISA_REGION leaves require a RegionManifest wrapper",
                    path=fragment_path,
                )
            fragment.validate_against(context.global_dag, fragment_path)
            leaf = fragment
        elif type(fragment) is RegionManifest:
            fragment.validate_against(context.global_dag, fragment_path)
            leaf = fragment.fragment
        else:
            raise SchemaError(
                "must be a CommandFragment or RegionManifest",
                path=fragment_path,
            )
        if (
            _requires_lifecycle_records(leaf, context)
            and not _has_lifecycle_records(leaf)
        ):
            raise SchemaError(
                "leaf is missing required fixed SRAM lifecycle records",
                path=f"{fragment_path}.fragment",
            )
        leaf_ids.append(leaf.id)
        claimed_action_ids.extend(leaf.claimed_action_ids)

    expected_leaf_ids = sorted(set(leaf_ids))
    if leaf_ids != expected_leaf_ids:
        raise SchemaError(
            "fragments must have unique leaves in canonical leaf-id order",
            path=path,
        )
    if len(claimed_action_ids) != len(set(claimed_action_ids)):
        raise SchemaError(
            "fragments must not overlap claimed actions",
            path=path,
        )
    expected_actions = {
        action.id
        for action in context.global_dag.actions
        if action.task_kind is not SemanticTaskKind.TRANSIT
    }
    if set(claimed_action_ids) != expected_actions:
        raise SchemaError(
            "fragments must exactly cover every executable global action",
            path=path,
        )


def _stage4_lowering_context(source: Stage4GlobalAction) -> LoweringContext:
    context = LoweringContext(
        ir1=source.graph,
        fusion_plans=source.fusion_plans,
        standalone_plans=source.standalone_plans,
        projection=source.projection,
        schedule_set=source.schedule_set,
        global_dag=source.global_dag,
    )
    context.validate("lowering_context")
    return context


@dataclass(frozen=True, slots=True)
class LoweredProgramProfile:
    """One profile's validated leaf fragments before manifest linking."""

    id: str
    source_global_action_entry_id: str
    source_scheduled_entry_id: str
    source_projected_entry_id: str
    source_planned_entry_id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    projection_context_id: str
    scheduling_context_id: str
    profile_id: str
    weight: float
    lowering_context: LoweringContext
    fragments: tuple[CommandFragment | RegionManifest, ...]

    @classmethod
    def create(
        cls,
        *,
        source: GlobalActionProfile,
        lowering_context: LoweringContext,
        fragments: tuple[CommandFragment | RegionManifest, ...],
    ) -> "LoweredProgramProfile":
        canonical_fragments = tuple(
            sorted(fragments, key=lambda fragment: _leaf_fragment(fragment).id)
        )
        semantic_key = {
            "source_global_action_entry_id": source.id,
            "source_scheduled_entry_id": source.source_scheduled_entry_id,
            "source_projected_entry_id": source.source_projected_entry_id,
            "source_planned_entry_id": source.source_planned_entry_id,
            "source_partitioned_entry_id": source.source_partitioned_entry_id,
            "source_ir1_id": source.source_ir1_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "lowering_context": lowering_context,
            "fragments": canonical_fragments,
        }
        return cls(
            id=stable_artifact_id(
                "lowered_program_profile",
                semantic_key,
                schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_global_action_entry_id": self.source_global_action_entry_id,
            "source_scheduled_entry_id": self.source_scheduled_entry_id,
            "source_projected_entry_id": self.source_projected_entry_id,
            "source_planned_entry_id": self.source_planned_entry_id,
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "lowering_context": self.lowering_context,
            "fragments": self.fragments,
        }

    def validate(self, path: str = "lowered_program_profile") -> None:
        for field_name in (
            "source_global_action_entry_id",
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
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if (
            self.source_ir1_id != self.lowering_context.ir1.id
            or self.profile_id
            != self.lowering_context.ir1.profile.stable_id()
        ):
            raise SchemaError(
                "must identify the embedded lowering profile exactly",
                path=path,
            )
        _validate_lowered_fragments(
            self.fragments,
            self.lowering_context,
            f"{path}.fragments",
        )
        expected_id = stable_artifact_id(
            "lowered_program_profile",
            self._semantic_key(),
            schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: GlobalActionProfile,
        path: str = "lowered_program_profile",
    ) -> None:
        source.validate("global_action_profile")
        self.validate(path)
        for field_name in (
            "source_scheduled_entry_id",
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
        if self.source_global_action_entry_id != source.id:
            raise SchemaError(
                "must match source global-action entry",
                path=f"{path}.source_global_action_entry_id",
            )
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError(
                "must preserve source profile and weight",
                path=path,
            )
        if self.lowering_context != source.lowering_context():
            raise SchemaError(
                "must preserve the complete source lowering context",
                path=f"{path}.lowering_context",
            )


@dataclass(frozen=True, slots=True)
class Stage4LoweredProgram:
    """One formal Stage 4 graph lowered to canonical leaf fragments."""

    schema_version: str
    producer_pass: str
    id: str
    source_global_action_carrier_id: str
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
    lowering_context: LoweringContext
    fragments: tuple[CommandFragment | RegionManifest, ...]

    @classmethod
    def create(
        cls,
        *,
        source: Stage4GlobalAction,
        fragments: tuple[CommandFragment | RegionManifest, ...],
    ) -> "Stage4LoweredProgram":
        canonical_fragments = tuple(
            sorted(fragments, key=lambda fragment: _leaf_fragment(fragment).id)
        )
        semantic_key = {
            "source_global_action_carrier_id": source.id,
            "source_scheduled_carrier_id": (
                source.source_scheduled_carrier_id
            ),
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
            "lowering_context": _stage4_lowering_context(source),
            "fragments": canonical_fragments,
        }
        return cls(
            schema_version=STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
            producer_pass="lowering",
            id=stable_artifact_id(
                "stage4_lowered_program",
                semantic_key,
                schema_version=STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_global_action_carrier_id": (
                self.source_global_action_carrier_id
            ),
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
            "lowering_context": self.lowering_context,
            "fragments": self.fragments,
        }

    def validate(self, path: str = "stage4_lowered_program") -> None:
        if self.schema_version != STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "lowering":
            raise SchemaError("must be 'lowering'", path=f"{path}.producer_pass")
        for field_name in (
            "source_global_action_carrier_id",
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
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if (
            self.lowering_context.ir1.producer_pass != "fusion_partition"
            or self.lowering_context.ir1.pd_plan_id != self.pd_plan.id
            or self.lowering_context.projection.producer_pass
            != "project_to_ir2"
            or self.lowering_context.schedule_set.producer_pass
            != "intra_die_schedule"
            or self.lowering_context.global_dag.producer_pass
            != "global_action_dag"
        ):
            raise SchemaError(
                "must preserve exact Stage 4 lowering provenance",
                path=f"{path}.lowering_context",
            )
        _validate_lowered_fragments(
            self.fragments,
            self.lowering_context,
            f"{path}.fragments",
        )
        expected_id = stable_artifact_id(
            "stage4_lowered_program",
            self._semantic_key(),
            schema_version=STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: Stage4GlobalAction,
        path: str = "stage4_lowered_program",
    ) -> None:
        if type(source) is not Stage4GlobalAction:
            raise SchemaError("must be a Stage4GlobalAction", path="source")
        source.validate("source")
        self.validate(path)
        if self.source_global_action_carrier_id != source.id:
            raise SchemaError(
                "must match source global-action carrier",
                path=f"{path}.source_global_action_carrier_id",
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
            "pd_plan",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve upstream Stage 4 provenance",
                    path=f"{path}.{field_name}",
                )
        if self.lowering_context != _stage4_lowering_context(source):
            raise SchemaError(
                "must preserve the complete source lowering context",
                path=f"{path}.lowering_context",
            )


@dataclass(frozen=True, slots=True)
class LoweredProgramBundle:
    """Ordered multi-profile lowering products before manifest linking."""

    schema_version: str
    producer_pass: str
    id: str
    source_global_action_bundle_id: str
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
    entries: tuple[LoweredProgramProfile, ...]

    @classmethod
    def create(
        cls,
        *,
        source: GlobalActionBundle,
        entries: tuple[LoweredProgramProfile, ...],
    ) -> "LoweredProgramBundle":
        semantic_key = {
            "source_global_action_bundle_id": source.id,
            "source_scheduled_bundle_id": source.source_scheduled_bundle_id,
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
            schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            producer_pass="lowering",
            id=stable_artifact_id(
                "lowered_program_bundle",
                semantic_key,
                schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_global_action_bundle_id": self.source_global_action_bundle_id,
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

    def validate(self, path: str = "lowered_program_bundle") -> None:
        if self.schema_version != LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "lowering":
            raise SchemaError("must be 'lowering'", path=f"{path}.producer_pass")
        for field_name in (
            "source_global_action_bundle_id",
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
        source_entry_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not LoweredProgramProfile:
                raise SchemaError(
                    "must be a LoweredProgramProfile",
                    path=entry_path,
                )
            entry.validate(entry_path)
            if (
                entry.profile_id != profile.profile_id
                or entry.lowering_context.ir1.profile != profile.key
            ):
                raise SchemaError(
                    "must exactly preserve source profile order",
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
            if entry.source_global_action_entry_id in source_entry_ids:
                raise SchemaError(
                    "duplicate source global-action entry",
                    path=f"{entry_path}.source_global_action_entry_id",
                )
            source_entry_ids.add(entry.source_global_action_entry_id)
            graph = entry.lowering_context.ir1
            if reference_fabric is None:
                reference_fabric = graph.fabric
                reference_groups = graph.groups
                reference_skeletons = graph.fused_op_skeletons
            elif (
                graph.fabric != reference_fabric
                or graph.groups != reference_groups
                or graph.fused_op_skeletons != reference_skeletons
            ):
                raise SchemaError(
                    "all profiles must share fabric, groups, and fusion partition",
                    path=f"{entry_path}.lowering_context.ir1",
                )
        expected_id = stable_artifact_id(
            "lowered_program_bundle",
            self._semantic_key(),
            schema_version=LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable bundle id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: GlobalActionBundle,
        path: str = "lowered_program_bundle",
    ) -> None:
        source.validate("global_action_bundle")
        self.validate(path)
        if self.source_global_action_bundle_id != source.id:
            raise SchemaError(
                "must match source global-action bundle",
                path=f"{path}.source_global_action_bundle_id",
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
                "must preserve the complete ordered profile manifest",
                path=f"{path}.source_profiles",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry.validate_against(
                source_entry,
                f"{path}.entries[{index}]",
            )


@dataclass(frozen=True, slots=True)
class LinkedProgramProfile:
    """One exact Lowered profile plus its linked manifest."""

    id: str
    source_lowered_entry_id: str
    source_global_action_entry_id: str
    source_scheduled_entry_id: str
    source_projected_entry_id: str
    source_planned_entry_id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    projection_context_id: str
    scheduling_context_id: str
    profile_id: str
    weight: float
    lowering_context: LoweringContext
    leaf_fragments: tuple[CommandFragment, ...]
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls,
        *,
        source: LoweredProgramProfile,
        manifest: LinkedProgramManifest,
    ) -> "LinkedProgramProfile":
        semantic_key = {
            "source_lowered_entry_id": source.id,
            "source_global_action_entry_id": source.source_global_action_entry_id,
            "source_scheduled_entry_id": source.source_scheduled_entry_id,
            "source_projected_entry_id": source.source_projected_entry_id,
            "source_planned_entry_id": source.source_planned_entry_id,
            "source_partitioned_entry_id": source.source_partitioned_entry_id,
            "source_ir1_id": source.source_ir1_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "lowering_context": source.lowering_context,
            "leaf_fragments": _leaf_fragments(source.fragments),
            "manifest": manifest,
        }
        return cls(
            id=stable_artifact_id(
                "linked_program_profile",
                semantic_key,
                schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_lowered_entry_id": self.source_lowered_entry_id,
            "source_global_action_entry_id": self.source_global_action_entry_id,
            "source_scheduled_entry_id": self.source_scheduled_entry_id,
            "source_projected_entry_id": self.source_projected_entry_id,
            "source_planned_entry_id": self.source_planned_entry_id,
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "projection_context_id": self.projection_context_id,
            "scheduling_context_id": self.scheduling_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "lowering_context": self.lowering_context,
            "leaf_fragments": self.leaf_fragments,
            "manifest": self.manifest,
        }

    def validate(self, path: str = "linked_program_profile") -> None:
        for field_name in (
            "source_lowered_entry_id",
            "source_global_action_entry_id",
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
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if (
            self.source_ir1_id != self.lowering_context.ir1.id
            or self.profile_id
            != self.lowering_context.ir1.profile.stable_id()
        ):
            raise SchemaError(
                "must identify the embedded lowering profile exactly",
                path=path,
            )
        if type(self.leaf_fragments) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.leaf_fragments")
        for index, fragment in enumerate(self.leaf_fragments):
            if type(fragment) is not CommandFragment:
                raise SchemaError(
                    "must be a leaf CommandFragment",
                    path=f"{path}.leaf_fragments[{index}]",
                )
        leaf_ids = tuple(fragment.id for fragment in self.leaf_fragments)
        if leaf_ids != tuple(sorted(set(leaf_ids))):
            raise SchemaError(
                "must contain unique leaf fragments in canonical id order",
                path=f"{path}.leaf_fragments",
            )
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError(
                "must be a LinkedProgramManifest",
                path=f"{path}.manifest",
            )
        if self.manifest.producer_pass != "manifest_linker":
            raise SchemaError(
                "must be produced by 'manifest_linker'",
                path=f"{path}.manifest.producer_pass",
            )
        if self.leaf_fragments != _leaf_fragments(self.manifest.fragments):
            raise SchemaError(
                "must exactly equal the manifest leaf fragments",
                path=f"{path}.leaf_fragments",
            )
        context = self.lowering_context
        self.manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            self.manifest.fragments,
            f"{path}.manifest",
        )
        expected_id = stable_artifact_id(
            "linked_program_profile",
            self._semantic_key(),
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: LoweredProgramProfile,
        path: str = "linked_program_profile",
    ) -> None:
        source.validate("lowered_program_profile")
        self.validate(path)
        for field_name in (
            "source_global_action_entry_id",
            "source_scheduled_entry_id",
            "source_projected_entry_id",
            "source_planned_entry_id",
            "source_partitioned_entry_id",
            "source_ir1_id",
            "projection_context_id",
            "scheduling_context_id",
            "profile_id",
            "weight",
            "lowering_context",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve lowered source provenance",
                    path=f"{path}.{field_name}",
                )
        if self.source_lowered_entry_id != source.id:
            raise SchemaError(
                "must match source lowered entry",
                path=f"{path}.source_lowered_entry_id",
            )
        if _fragments_by_leaf_id(
            self.manifest.fragments
        ) != _fragments_by_leaf_id(source.fragments):
            raise SchemaError(
                "manifest must bijectively consume the exact lowered fragments",
                path=f"{path}.manifest.fragments",
            )
        if self.leaf_fragments != _leaf_fragments(source.fragments):
            raise SchemaError(
                "must preserve the exact lowered leaves",
                path=f"{path}.leaf_fragments",
            )


@dataclass(frozen=True, slots=True)
class Stage4LinkedProgram:
    """One formal Stage 4 lowering product plus its linked manifest."""

    schema_version: str
    producer_pass: str
    id: str
    source_lowered_carrier_id: str
    source_global_action_carrier_id: str
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
    lowering_context: LoweringContext
    leaf_fragments: tuple[CommandFragment, ...]
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls,
        *,
        source: Stage4LoweredProgram,
        manifest: LinkedProgramManifest,
    ) -> "Stage4LinkedProgram":
        semantic_key = {
            "source_lowered_carrier_id": source.id,
            "source_global_action_carrier_id": (
                source.source_global_action_carrier_id
            ),
            "source_scheduled_carrier_id": (
                source.source_scheduled_carrier_id
            ),
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
            "lowering_context": source.lowering_context,
            "leaf_fragments": _leaf_fragments(source.fragments),
            "manifest": manifest,
        }
        return cls(
            schema_version=STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="manifest_linker",
            id=stable_artifact_id(
                "stage4_linked_program",
                semantic_key,
                schema_version=STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_lowered_carrier_id": self.source_lowered_carrier_id,
            "source_global_action_carrier_id": (
                self.source_global_action_carrier_id
            ),
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
            "lowering_context": self.lowering_context,
            "leaf_fragments": self.leaf_fragments,
            "manifest": self.manifest,
        }

    def validate(self, path: str = "stage4_linked_program") -> None:
        if self.schema_version != STAGE4_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "manifest_linker":
            raise SchemaError(
                "must be 'manifest_linker'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_lowered_carrier_id",
            "source_global_action_carrier_id",
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
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if (
            self.lowering_context.ir1.producer_pass != "fusion_partition"
            or self.lowering_context.ir1.pd_plan_id != self.pd_plan.id
            or self.lowering_context.projection.producer_pass
            != "project_to_ir2"
            or self.lowering_context.schedule_set.producer_pass
            != "intra_die_schedule"
            or self.lowering_context.global_dag.producer_pass
            != "global_action_dag"
        ):
            raise SchemaError(
                "must preserve exact Stage 4 lowering provenance",
                path=f"{path}.lowering_context",
            )
        if type(self.leaf_fragments) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.leaf_fragments",
            )
        for index, fragment in enumerate(self.leaf_fragments):
            if type(fragment) is not CommandFragment:
                raise SchemaError(
                    "must be a leaf CommandFragment",
                    path=f"{path}.leaf_fragments[{index}]",
                )
        leaf_ids = tuple(fragment.id for fragment in self.leaf_fragments)
        if leaf_ids != tuple(sorted(set(leaf_ids))):
            raise SchemaError(
                "must contain unique leaf fragments in canonical id order",
                path=f"{path}.leaf_fragments",
            )
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError(
                "must be a LinkedProgramManifest",
                path=f"{path}.manifest",
            )
        if self.manifest.producer_pass != "manifest_linker":
            raise SchemaError(
                "must be produced by manifest_linker",
                path=f"{path}.manifest.producer_pass",
            )
        if self.leaf_fragments != _leaf_fragments(self.manifest.fragments):
            raise SchemaError(
                "must exactly equal the manifest leaf fragments",
                path=f"{path}.leaf_fragments",
            )
        context = self.lowering_context
        self.manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            self.manifest.fragments,
            f"{path}.manifest",
        )
        expected_id = stable_artifact_id(
            "stage4_linked_program",
            self._semantic_key(),
            schema_version=STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: Stage4LoweredProgram,
        path: str = "stage4_linked_program",
    ) -> None:
        if type(source) is not Stage4LoweredProgram:
            raise SchemaError("must be a Stage4LoweredProgram", path="source")
        source.validate("source")
        self.validate(path)
        if self.source_lowered_carrier_id != source.id:
            raise SchemaError(
                "must match source lowered carrier",
                path=f"{path}.source_lowered_carrier_id",
            )
        for field_name in (
            "source_global_action_carrier_id",
            "source_scheduled_carrier_id",
            "source_projected_carrier_id",
            "source_planned_carrier_id",
            "source_partitioned_carrier_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
            "pd_plan",
            "lowering_context",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve lowered Stage 4 provenance",
                    path=f"{path}.{field_name}",
                )
        if _fragments_by_leaf_id(
            self.manifest.fragments
        ) != _fragments_by_leaf_id(source.fragments):
            raise SchemaError(
                "manifest must bijectively consume exact lowered fragments",
                path=f"{path}.manifest.fragments",
            )
        if self.leaf_fragments != _leaf_fragments(source.fragments):
            raise SchemaError(
                "must preserve exact lowered leaves",
                path=f"{path}.leaf_fragments",
            )


@dataclass(frozen=True, slots=True)
class LinkedProgramBundle:
    """Ordered multi-profile link products with complete lowering provenance."""

    schema_version: str
    producer_pass: str
    id: str
    source_lowered_bundle_id: str
    source_global_action_bundle_id: str
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
    entries: tuple[LinkedProgramProfile, ...]

    @classmethod
    def create(
        cls,
        *,
        source: LoweredProgramBundle,
        entries: tuple[LinkedProgramProfile, ...],
    ) -> "LinkedProgramBundle":
        semantic_key = {
            "source_lowered_bundle_id": source.id,
            "source_global_action_bundle_id": source.source_global_action_bundle_id,
            "source_scheduled_bundle_id": source.source_scheduled_bundle_id,
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
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            producer_pass="manifest_linker",
            id=stable_artifact_id(
                "linked_program_bundle",
                semantic_key,
                schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_lowered_bundle_id": self.source_lowered_bundle_id,
            "source_global_action_bundle_id": self.source_global_action_bundle_id,
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

    def validate(self, path: str = "linked_program_bundle") -> None:
        if self.schema_version != LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "manifest_linker":
            raise SchemaError(
                "must be 'manifest_linker'",
                path=f"{path}.producer_pass",
            )
        for field_name in (
            "source_lowered_bundle_id",
            "source_global_action_bundle_id",
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
        source_entry_ids: set[str] = set()
        source_global_entry_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not LinkedProgramProfile:
                raise SchemaError(
                    "must be a LinkedProgramProfile",
                    path=entry_path,
                )
            entry.validate(entry_path)
            if (
                entry.profile_id != profile.profile_id
                or entry.lowering_context.ir1.profile != profile.key
            ):
                raise SchemaError(
                    "must exactly preserve source profile order",
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
            if entry.source_lowered_entry_id in source_entry_ids:
                raise SchemaError(
                    "duplicate source lowered entry",
                    path=f"{entry_path}.source_lowered_entry_id",
                )
            source_entry_ids.add(entry.source_lowered_entry_id)
            if entry.source_global_action_entry_id in source_global_entry_ids:
                raise SchemaError(
                    "duplicate source global-action entry",
                    path=f"{entry_path}.source_global_action_entry_id",
                )
            source_global_entry_ids.add(entry.source_global_action_entry_id)
            graph = entry.lowering_context.ir1
            if reference_fabric is None:
                reference_fabric = graph.fabric
                reference_groups = graph.groups
                reference_skeletons = graph.fused_op_skeletons
            elif (
                graph.fabric != reference_fabric
                or graph.groups != reference_groups
                or graph.fused_op_skeletons != reference_skeletons
            ):
                raise SchemaError(
                    "all profiles must share fabric, groups, and fusion partition",
                    path=f"{entry_path}.lowering_context.ir1",
                )
        expected_id = stable_artifact_id(
            "linked_program_bundle",
            self._semantic_key(),
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable bundle id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: LoweredProgramBundle,
        path: str = "linked_program_bundle",
    ) -> None:
        source.validate("lowered_program_bundle")
        self.validate(path)
        if self.source_lowered_bundle_id != source.id:
            raise SchemaError(
                "must match source lowered bundle",
                path=f"{path}.source_lowered_bundle_id",
            )
        for field_name in (
            "source_global_action_bundle_id",
            "source_scheduled_bundle_id",
            "source_projected_bundle_id",
            "source_inter_die_bundle_id",
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
            "projection_context_id",
            "scheduling_context_id",
            "source_profiles",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve lowered bundle provenance",
                    path=f"{path}.{field_name}",
                )
        if len(self.entries) != len(source.entries):
            raise SchemaError(
                "must preserve the complete ordered profile manifest",
                path=f"{path}.entries",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry.validate_against(source_entry, f"{path}.entries[{index}]")


__all__ = [
    "LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION",
    "LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION",
    "STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION",
    "STAGE4_LINKED_PROGRAM_SCHEMA_VERSION",
    "LoweredProgramProfile",
    "LoweredProgramBundle",
    "LinkedProgramProfile",
    "LinkedProgramBundle",
    "Stage4LoweredProgram",
    "Stage4LinkedProgram",
]
