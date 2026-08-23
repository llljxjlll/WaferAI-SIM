"""Shared deterministic placement for semantic MoE work.

The first layer deliberately knows nothing about planner task ids or DAG order.
Every consumer (candidate scheduling, resource costing, and IR2 lowering) must
present the complete semantic work set for a rank, so one canonical round-robin
decides ownership everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .ir0 import FusionPattern
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir1 import IR1
from .serde import canonical_digest
from .swizzle import SwizzleActionKind, SwizzleAlgorithm
if TYPE_CHECKING:
    from .swizzle_moe import (
        MoeActionWitness, MoeHardwareFacts, MoeSwizzleProblem,
    )


_DOMAINS = {
    "dispatch_source_token",
    "dispatch_expert_tile",
    "combine_down_tile",
    "combine_source_token",
    "comp",
}


@dataclass(frozen=True, slots=True)
class MoeSemanticWorkKey:
    domain: str
    token_index: int | None
    expert_index: int | None
    tile_index: int | None
    n_block: int | None
    role: str
    placement_group_index: int | None = None

    def validate(self, path: str = "moe_semantic_work_key") -> None:
        if self.domain not in _DOMAINS:
            raise SchemaError("unsupported semantic work domain", path=f"{path}.domain")
        validate_nonempty(self.role, f"{path}.role")
        for name in ("token_index", "expert_index", "tile_index", "n_block", "placement_group_index"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if self.domain.endswith("source_token") and self.token_index is None:
            raise SchemaError("source-token work requires token_index", path=path)
        if self.domain in ("dispatch_expert_tile", "combine_down_tile", "comp") and (
            self.expert_index is None or self.tile_index is None
        ):
            raise SchemaError("expert work requires expert/tile indices", path=path)
        if self.domain == "comp" and self.n_block is None:
            raise SchemaError("COMP work requires n_block", path=path)

    @property
    def sort_key(self) -> tuple[object, ...]:
        number = lambda value: -1 if value is None else value
        return (
            self.domain,
            number(self.token_index),
            number(self.expert_index),
            number(self.tile_index),
            number(self.n_block),
            number(self.placement_group_index),
            self.role,
        )


@dataclass(frozen=True, slots=True)
class MoeSemanticWorkItem:
    rank: int
    work_key: MoeSemanticWorkKey

    def validate(self, path: str = "moe_semantic_work_item") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        self.work_key.validate(f"{path}.work_key")


@dataclass(frozen=True, slots=True)
class MoeWorkOwner:
    rank: int
    work_key: MoeSemanticWorkKey
    logical_core: LogicalCoreRef
    runtime_core_id: int

    def validate(self, path: str = "moe_work_owner") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        self.work_key.validate(f"{path}.work_key")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.logical_core.die_id != self.rank:
            raise SchemaError("owner core must reside on the work rank", path=path)


@dataclass(frozen=True, slots=True)
class MoeWorkloadActionOwner:
    action_ref: str
    rank: int
    logical_core: LogicalCoreRef
    runtime_core_id: int
    placement_group_index: int
    owner_witness_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.action_ref, f"{path}.action_ref")
        for name in ("rank", "runtime_core_id", "placement_group_index"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        self.logical_core.validate(f"{path}.logical_core")
        if self.logical_core.die_id != self.rank:
            raise SchemaError("whole-workload owner must reside on action rank", path=path)
        if not self.owner_witness_refs or len(self.owner_witness_refs) != len(set(self.owner_witness_refs)):
            raise SchemaError("whole-workload owner witnesses must be unique/nonempty", path=f"{path}.owner_witness_refs")
        for index, ref in enumerate(self.owner_witness_refs):
            validate_nonempty(ref, f"{path}.owner_witness_refs[{index}]")


@dataclass(frozen=True, slots=True)
class MoeEndpointCoreWidth:
    logical_core: LogicalCoreRef
    runtime_core_id: int
    session_action_refs: tuple[str, ...]
    max_inflight: int

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.max_inflight, f"{path}.max_inflight")
        if not self.session_action_refs or self.max_inflight == 0 or self.max_inflight > len(self.session_action_refs):
            raise SchemaError("endpoint width witness is invalid", path=path)
        if len(self.session_action_refs) != len(set(self.session_action_refs)):
            raise SchemaError(
                "endpoint session refs are duplicated",
                path=f"{path}.session_action_refs",
            )


MOE_WORKLOAD_ENDPOINT_FEASIBILITY_SCHEMA_VERSION = (
    "wafer_frontend.moe_workload_endpoint_feasibility/v1alpha1"
)
MOE_WHOLE_PAIR_FEASIBILITY_SCHEMA_VERSION = (
    "wafer_frontend.moe_whole_pair_feasibility/v1alpha1"
)
MOE_WHOLE_PAIR_PLACEMENT_FEASIBILITY_SCHEMA_VERSION = (
    "wafer_frontend.moe_whole_pair_placement_feasibility/v1alpha1"
)


class MoeWholePairPlacementReason(str, Enum):
    ADMITTED = "admitted"
    ORDINARY_VALUE_CROSS_CORE = "ordinary_value_cross_core"
    PRESERVED_OWNER_MISMATCH = "preserved_owner_mismatch"


_PLACEMENT_FAILURE_DIAGNOSTICS = {
    MoeWholePairPlacementReason.PRESERVED_OWNER_MISMATCH: (
        "build_moe_swizzle_workload_placement.workload_projection",
        "preserved action spans multiple M-block owners without LOCAL_COPY",
    ),
}


@dataclass(frozen=True, slots=True)
class MoeWholePairPlacementFeasibility:
    schema_version: str
    producer_pass: str
    id: str
    workload_projection_id: str
    replacement_projection_id: str
    placement_digest: str | None
    reason: MoeWholePairPlacementReason
    failure_path: str | None
    failure_message: str | None
    feasible: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeWholePairPlacementFeasibility":
        result = cls(
            MOE_WHOLE_PAIR_PLACEMENT_FEASIBILITY_SCHEMA_VERSION,
            "build_moe_swizzle_whole_pair_placement_feasibility",
            stable_artifact_id(
                "moe_whole_pair_placement_feasibility", semantic,
                schema_version=MOE_WHOLE_PAIR_PLACEMENT_FEASIBILITY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_whole_pair_placement_feasibility") -> None:
        if (
            self.schema_version
            != MOE_WHOLE_PAIR_PLACEMENT_FEASIBILITY_SCHEMA_VERSION
            or self.producer_pass
            != "build_moe_swizzle_whole_pair_placement_feasibility"
        ):
            raise SchemaError("unsupported pair placement schema/producer", path=path)
        validate_nonempty(self.workload_projection_id, f"{path}.workload_projection_id")
        validate_nonempty(self.replacement_projection_id, f"{path}.replacement_projection_id")
        if type(self.reason) is not MoeWholePairPlacementReason or type(self.feasible) is not bool:
            raise SchemaError("placement result must be typed", path=path)
        expected_feasible = self.reason is MoeWholePairPlacementReason.ADMITTED
        if self.feasible != expected_feasible:
            raise SchemaError("placement feasible flag disagrees with reason", path=f"{path}.feasible")
        if self.placement_digest is not None and len(self.placement_digest) != 64:
            raise SchemaError("placement digest must be sha256 when present", path=f"{path}.placement_digest")
        if expected_feasible:
            if (
                self.placement_digest is None
                or self.failure_path is not None
                or self.failure_message is not None
            ):
                raise SchemaError("admitted placement cannot carry a failure", path=path)
        else:
            exact = _PLACEMENT_FAILURE_DIAGNOSTICS.get(self.reason)
            ordinary_cross_core = (
                self.reason is MoeWholePairPlacementReason.ORDINARY_VALUE_CROSS_CORE
                and (
                    (
                        self.failure_path == "projection.values"
                        and self.failure_message
                        == "value crosses cores without an explicit LOCAL_COPY"
                    )
                    or (
                        isinstance(self.failure_path, str)
                        and self.failure_path.startswith("value_bridge.bindings[")
                        and isinstance(self.failure_message, str)
                        and self.failure_message.startswith(
                            "value bridge lacks physical root "
                        )
                        and "linked_owners=" in self.failure_message
                    )
                    or (
                        isinstance(self.failure_path, str)
                        and self.failure_path.startswith(
                            "schedule_moe_swizzle_workload_storage_reuse.bindings["
                        )
                        and self.failure_message
                        == "SWIGLU occupant crosses cores without LOCAL_COPY"
                    )
                )
            )
            if not ordinary_cross_core and (
                exact is None
                or (self.failure_path, self.failure_message) != exact
            ):
                raise SchemaError("placement failure diagnostic is not exact", path=path)
        semantic = {name: getattr(self, name) for name in (
            "workload_projection_id", "replacement_projection_id",
            "placement_digest", "reason", "failure_path", "failure_message",
            "feasible",
        )}
        if self.id != stable_artifact_id(
            "moe_whole_pair_placement_feasibility", semantic,
            schema_version=MOE_WHOLE_PAIR_PLACEMENT_FEASIBILITY_SCHEMA_VERSION,
        ):
            raise SchemaError("unstable pair placement id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeWorkloadEndpointFeasibility:
    schema_version: str
    producer_pass: str
    id: str
    workload_projection_id: str
    replacement_projection_id: str
    placement_digest: str
    widths: tuple[MoeEndpointCoreWidth, ...]
    capacity_per_core: int
    peak_width: int
    feasible: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeWorkloadEndpointFeasibility":
        result = cls(
            MOE_WORKLOAD_ENDPOINT_FEASIBILITY_SCHEMA_VERSION,
            "build_moe_swizzle_workload_endpoint_feasibility",
            stable_artifact_id(
                "moe_workload_endpoint_feasibility", semantic,
                schema_version=MOE_WORKLOAD_ENDPOINT_FEASIBILITY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_workload_endpoint_feasibility") -> None:
        if (
            self.schema_version != MOE_WORKLOAD_ENDPOINT_FEASIBILITY_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_workload_endpoint_feasibility"
        ):
            raise SchemaError("unsupported endpoint feasibility schema/producer", path=path)
        for name in ("workload_projection_id", "replacement_projection_id", "placement_digest"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if len(self.placement_digest) != 64:
            raise SchemaError("placement digest must be sha256", path=f"{path}.placement_digest")
        validate_uint64(self.capacity_per_core, f"{path}.capacity_per_core")
        validate_uint64(self.peak_width, f"{path}.peak_width")
        if not self.widths or self.capacity_per_core == 0:
            raise SchemaError("endpoint witness requires widths/capacity", path=path)
        for index, item in enumerate(self.widths):
            item.validate(f"{path}.widths[{index}]")
        if self.peak_width != max(item.max_inflight for item in self.widths):
            raise SchemaError("endpoint peak does not match per-core widths", path=path)
        if type(self.feasible) is not bool or self.feasible != (self.peak_width <= self.capacity_per_core):
            raise SchemaError("endpoint feasible flag is not exact", path=f"{path}.feasible")
        semantic = {
            "workload_projection_id": self.workload_projection_id,
            "replacement_projection_id": self.replacement_projection_id,
            "placement_digest": self.placement_digest,
            "widths": self.widths,
            "capacity_per_core": self.capacity_per_core,
            "peak_width": self.peak_width,
            "feasible": self.feasible,
        }
        if self.id != stable_artifact_id(
            "moe_workload_endpoint_feasibility", semantic,
            schema_version=MOE_WORKLOAD_ENDPOINT_FEASIBILITY_SCHEMA_VERSION,
        ):
            raise SchemaError("unstable endpoint feasibility id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeStorageIntervalColorDepth:
    runtime_core_id: int
    family: str
    depth: int

    def validate(self, path: str) -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.depth, f"{path}.depth")
        if self.family != "swiglu_output" or self.depth not in (1, 2):
            raise SchemaError("storage interval color depth is invalid", path=path)


@dataclass(frozen=True, slots=True)
class MoeWholeCoreLifecycleCount:
    runtime_core_id: int
    alloc_count: int
    bind_count: int
    free_count: int

    def validate(self, path: str) -> None:
        for name in (
            "runtime_core_id", "alloc_count", "bind_count", "free_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.free_count > self.alloc_count:
            raise SchemaError("per-core FREE exceeds ALLOC", path=path)


@dataclass(frozen=True, slots=True)
class MoeCandidateCoreLifecycleFloor:
    """Safe per-region lifecycle floor derived without whole-pair lowering."""

    candidate_ref: str
    runtime_core_id: int
    alloc_count: int
    bind_count: int
    free_count: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.candidate_ref, f"{path}.candidate_ref")
        for name in (
            "runtime_core_id", "alloc_count", "bind_count", "free_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.free_count > self.alloc_count
            or self.alloc_count + self.bind_count + self.free_count == 0
        ):
            raise SchemaError("candidate lifecycle floor is invalid", path=path)


@dataclass(frozen=True, slots=True)
class MoeWholePairFeasibility:
    schema_version: str
    producer_pass: str
    id: str
    candidate_refs: tuple[str, str]
    placement: MoeWholePairPlacementFeasibility
    endpoint: MoeWorkloadEndpointFeasibility | None
    workload_abi_id: str | None
    dynamic_root_keys: tuple[tuple[int, str, int], ...]
    dynamic_sram_high_water_bytes: int
    dynamic_sram_capacity_bytes: int
    storage_color_depths: tuple[MoeStorageIntervalColorDepth, ...]
    core_lifecycle_counts: tuple[MoeWholeCoreLifecycleCount, ...]
    whole_physical_root_count: int
    whole_alloc_count: int
    whole_free_count: int
    feasible: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeWholePairFeasibility":
        result = cls(
            MOE_WHOLE_PAIR_FEASIBILITY_SCHEMA_VERSION,
            "build_moe_swizzle_whole_pair_feasibility",
            stable_artifact_id(
                "moe_whole_pair_feasibility", semantic,
                schema_version=MOE_WHOLE_PAIR_FEASIBILITY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "moe_whole_pair_feasibility") -> None:
        if (
            self.schema_version != MOE_WHOLE_PAIR_FEASIBILITY_SCHEMA_VERSION
            or self.producer_pass
            != "build_moe_swizzle_whole_pair_feasibility"
        ):
            raise SchemaError("unsupported whole-pair feasibility schema/producer", path=path)
        if len(set(self.candidate_refs)) != 2:
            raise SchemaError("whole pair requires ordered distinct candidate refs", path=f"{path}.candidate_refs")
        if type(self.placement) is not MoeWholePairPlacementFeasibility:
            raise SchemaError("whole pair requires exact placement witness", path=f"{path}.placement")
        self.placement.validate(f"{path}.placement")
        if self.dynamic_root_keys != tuple(sorted(set(self.dynamic_root_keys))):
            raise SchemaError("dynamic root keys must be canonical/unique", path=f"{path}.dynamic_root_keys")
        validate_uint64(self.dynamic_sram_high_water_bytes, f"{path}.dynamic_sram_high_water_bytes")
        validate_uint64(self.dynamic_sram_capacity_bytes, f"{path}.dynamic_sram_capacity_bytes")
        for name in (
            "whole_physical_root_count", "whole_alloc_count", "whole_free_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            type(self.storage_color_depths) is not tuple
            or self.storage_color_depths != tuple(sorted(
                self.storage_color_depths,
                key=lambda item: (item.runtime_core_id, item.family),
            ))
            or len({(item.runtime_core_id, item.family) for item in self.storage_color_depths})
            != len(self.storage_color_depths)
        ):
            raise SchemaError("storage color depths must be canonical/unique", path=f"{path}.storage_color_depths")
        for index, item in enumerate(self.storage_color_depths):
            if type(item) is not MoeStorageIntervalColorDepth:
                raise SchemaError("requires exact storage color depth", path=f"{path}.storage_color_depths[{index}]")
            item.validate(f"{path}.storage_color_depths[{index}]")
        if (
            type(self.core_lifecycle_counts) is not tuple
            or self.core_lifecycle_counts != tuple(sorted(
                self.core_lifecycle_counts,
                key=lambda item: item.runtime_core_id,
            ))
            or len({item.runtime_core_id for item in self.core_lifecycle_counts})
            != len(self.core_lifecycle_counts)
        ):
            raise SchemaError(
                "per-core lifecycle counts must be canonical/unique",
                path=f"{path}.core_lifecycle_counts",
            )
        for index, item in enumerate(self.core_lifecycle_counts):
            if type(item) is not MoeWholeCoreLifecycleCount:
                raise SchemaError(
                    "requires exact per-core lifecycle count",
                    path=f"{path}.core_lifecycle_counts[{index}]",
                )
            item.validate(f"{path}.core_lifecycle_counts[{index}]")
        if self.placement.feasible:
            if type(self.endpoint) is not MoeWorkloadEndpointFeasibility:
                raise SchemaError("placed pair requires endpoint witness", path=f"{path}.endpoint")
            self.endpoint.validate(f"{path}.endpoint")
            if (
                self.endpoint.workload_projection_id
                != self.placement.workload_projection_id
                or self.endpoint.replacement_projection_id
                != self.placement.replacement_projection_id
                or self.endpoint.placement_digest
                != self.placement.placement_digest
            ):
                raise SchemaError(
                    "placement and endpoint lineage do not close",
                    path=path,
                )
            validate_nonempty(self.workload_abi_id, f"{path}.workload_abi_id")
            if (
                not self.storage_color_depths or not self.core_lifecycle_counts
                or self.whole_physical_root_count == 0
                or self.whole_alloc_count > self.whole_physical_root_count
                or self.whole_free_count > self.whole_alloc_count
                or sum(item.alloc_count for item in self.core_lifecycle_counts)
                != self.whole_alloc_count
                or sum(item.free_count for item in self.core_lifecycle_counts)
                != self.whole_free_count
            ):
                raise SchemaError("placed pair whole lifecycle truth is invalid", path=path)
            expected_feasible = (
                self.endpoint.feasible
                and self.dynamic_sram_high_water_bytes
                <= self.dynamic_sram_capacity_bytes
            )
        else:
            if (
                self.endpoint is not None
                or self.workload_abi_id is not None
                or self.dynamic_root_keys
                or self.dynamic_sram_high_water_bytes != 0
                or self.storage_color_depths
                or self.core_lifecycle_counts
                or self.whole_physical_root_count != 0
                or self.whole_alloc_count != 0
                or self.whole_free_count != 0
            ):
                raise SchemaError(
                    "placement-infeasible pair cannot fabricate endpoint/ABI/root truth",
                    path=path,
                )
            expected_feasible = False
        if type(self.feasible) is not bool or self.feasible != expected_feasible:
            raise SchemaError("whole-pair feasible flag is not exact", path=f"{path}.feasible")
        semantic = {name: getattr(self, name) for name in (
            "candidate_refs", "placement", "endpoint", "workload_abi_id",
            "dynamic_root_keys", "dynamic_sram_high_water_bytes", "dynamic_sram_capacity_bytes", "feasible",
            "storage_color_depths", "core_lifecycle_counts", "whole_physical_root_count",
            "whole_alloc_count", "whole_free_count",
        )}
        if self.id != stable_artifact_id(
            "moe_whole_pair_feasibility", semantic,
            schema_version=MOE_WHOLE_PAIR_FEASIBILITY_SCHEMA_VERSION,
        ):
            raise SchemaError("unstable whole-pair feasibility id", path=f"{path}.id")


def _owner_affinity(key: MoeSemanticWorkKey) -> tuple[object, ...]:
    # Source-token transport remains source-token owned even when its typed
    # pipeline provenance names an expert M-block.  Treating that provenance
    # as placement affinity makes the same candidate move cores when another
    # fusion region is added to the complete work set.
    if key.domain.endswith("source_token"):
        return ("source_token", key.token_index)
    if key.placement_group_index is not None and key.expert_index is not None:
        return ("expert_m_block", key.expert_index, key.placement_group_index)
    return (
        "expert_tile",
        key.expert_index,
        key.tile_index,
        0 if key.n_block is None else key.n_block,
    )


def _owner_round_robin_ordinal(key: MoeSemanticWorkKey) -> int:
    """Return a stable typed RR ordinal independent of surrounding work."""

    affinity = _owner_affinity(key)
    if affinity[0] == "source_token":
        assert key.token_index is not None
        return key.token_index
    if affinity[0] == "expert_m_block":
        assert key.placement_group_index is not None
        return key.placement_group_index
    if key.tile_index is None:
        raise SchemaError(
            "expert-tile affinity lacks a stable tile ordinal",
            path="build_moe_work_owner_map.semantic_work_items",
        )
    return key.tile_index


def build_moe_work_owner_map(
    hardware_facts: MoeHardwareFacts,
    semantic_work_items: tuple[MoeSemanticWorkItem, ...],
) -> tuple[MoeWorkOwner, ...]:
    """Assign the complete unique work-key set per rank over ordered real cores."""

    hardware_facts.validate("build_moe_work_owner_map.hardware_facts")
    if type(semantic_work_items) is not tuple or not semantic_work_items:
        raise SchemaError("semantic work set must be a nonempty tuple", path="build_moe_work_owner_map.semantic_work_items")
    unique: dict[tuple[int, MoeSemanticWorkKey], MoeSemanticWorkItem] = {}
    for index, item in enumerate(semantic_work_items):
        if type(item) is not MoeSemanticWorkItem:
            raise SchemaError("requires exact MoeSemanticWorkItem", path=f"build_moe_work_owner_map.semantic_work_items[{index}]")
        item.validate(f"build_moe_work_owner_map.semantic_work_items[{index}]")
        unique[(item.rank, item.work_key)] = item
    owners = []
    ranks = sorted({rank for rank, _ in unique})
    for rank in ranks:
        if rank >= len(hardware_facts.ordered_cores_by_die):
            raise SchemaError("work rank has no hardware-facts die", path="build_moe_work_owner_map.semantic_work_items")
        cores = hardware_facts.ordered_cores_by_die[rank]
        if not cores:
            raise SchemaError("work rank has no schedulable core", path="build_moe_work_owner_map.hardware_facts")
        keys = sorted((key for item_rank, key in unique if item_rank == rank), key=lambda item: item.sort_key)
        for key in keys:
            core = cores[_owner_round_robin_ordinal(key) % len(cores)]
            owners.append(MoeWorkOwner(rank, key, core.logical_core, core.runtime_core_id))
    owners.sort(key=lambda item: (item.rank, item.work_key.sort_key))
    result = tuple(owners)
    for index, owner in enumerate(result):
        owner.validate(f"build_moe_work_owner_map.result[{index}]")
    return result


def _candidate_action_work_key(
    problem: MoeSwizzleProblem,
    action: MoeActionWitness,
) -> MoeSemanticWorkKey:
    placement_group = (
        action.tile_index
        if len(action.assignment_refs) == 1
        else action.pipeline_index
    )
    if action.kind is SwizzleActionKind.COMP:
        return MoeSemanticWorkKey(
            "comp", None, action.expert_index, action.tile_index,
            0 if action.n_block_index is None else action.n_block_index,
            action.work_role,
            placement_group,
        )
    expert_side = action.rank == action.expert_index
    dispatch = problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
    if dispatch:
        domain = "dispatch_expert_tile" if expert_side else "dispatch_source_token"
    else:
        domain = "combine_down_tile" if expert_side else "combine_source_token"
    return MoeSemanticWorkKey(
        domain,
        action.tile_index if domain.endswith("source_token") else None,
        action.expert_index,
        action.tile_index,
        action.n_block_index if domain == "combine_down_tile" else None,
        action.work_role,
        placement_group,
    )


def build_moe_candidate_action_owner_map(
    problem: MoeSwizzleProblem,
    actions: tuple[MoeActionWitness, ...],
) -> dict[str, MoeWorkOwner]:
    """Bind a complete candidate action set through the canonical work owners."""

    from .swizzle_moe import MoeActionWitness

    problem.validate("build_moe_candidate_action_owner_map.problem")
    if type(actions) is not tuple or not actions:
        raise SchemaError("candidate action set must be a nonempty tuple", path="build_moe_candidate_action_owner_map.actions")
    keys = {}
    for index, action in enumerate(actions):
        if type(action) is not MoeActionWitness:
            raise SchemaError("requires exact MoeActionWitness", path=f"build_moe_candidate_action_owner_map.actions[{index}]")
        action.validate(f"build_moe_candidate_action_owner_map.actions[{index}]")
        if action.id in keys:
            raise SchemaError("candidate action id is duplicated", path="build_moe_candidate_action_owner_map.actions")
        keys[action.id] = _candidate_action_work_key(problem, action)
    owners = {
        (item.rank, item.work_key): item
        for item in build_moe_work_owner_map(
            problem.hardware_facts,
            tuple(MoeSemanticWorkItem(action.rank, keys[action.id]) for action in actions),
        )
    }
    return {action.id: owners[(action.rank, keys[action.id])] for action in actions}


def _task_work_key(task: object) -> MoeSemanticWorkKey:
    role = getattr(task, "work_role", None)
    if type(role) is not str or not role:
        raise SchemaError("IR2 task lacks typed work_role", path="build_moe_swizzle_task_placement.projection.tasks")
    token = task.tile_index
    if task.kind is SwizzleActionKind.COMP:
        return MoeSemanticWorkKey(
            "comp", None, task.expert_index, task.tile_index,
            0 if task.n_block is None else task.n_block, role,
            task.pipeline_index,
        )
    if task.kind not in (
        SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT,
    ):
        domain = "dispatch_expert_tile"
    elif role not in (
        "moe_dispatch_gemm.transport",
        "moe_gemm_combine.transport",
        "transport",
    ):
        raise SchemaError("transport task lacks typed fusion-region role", path="build_moe_swizzle_task_placement.projection.tasks")
    elif role == "transport":
        is_expert = task.rank == task.expert_index
        domain = "combine_down_tile" if task.kind is SwizzleActionKind.SEND and is_expert else "dispatch_source_token" if task.kind is SwizzleActionKind.SEND else "dispatch_expert_tile" if is_expert else "combine_source_token"
    elif task.kind is SwizzleActionKind.SEND:
        domain = (
            "dispatch_source_token"
            if role == "moe_dispatch_gemm.transport"
            else "combine_down_tile"
        )
    elif task.kind in (SwizzleActionKind.RECV, SwizzleActionKind.WAIT):
        domain = (
            "dispatch_expert_tile"
            if role == "moe_dispatch_gemm.transport"
            else "combine_source_token"
        )
    return MoeSemanticWorkKey(
        domain,
        token if domain.endswith("source_token") else None,
        task.expert_index,
        task.tile_index,
        task.n_block if domain in ("combine_down_tile", "comp") else None,
        role,
        task.pipeline_index,
    )


def build_moe_swizzle_task_placement(
    ir1: IR1,
    projection: object,
    hardware_facts: MoeHardwareFacts,
) -> tuple[tuple[str, MoeWorkOwner], ...]:
    """Cross-check IR1/facts and bind every projected task via shared owners."""

    ir1.validate("build_moe_swizzle_task_placement.ir1")
    projection.validate("build_moe_swizzle_task_placement.projection")
    hardware_facts.validate("build_moe_swizzle_task_placement.hardware_facts")
    if hardware_facts.source_fabric_digest != canonical_digest(ir1.fabric):
        raise SchemaError("hardware facts belong to another IR1 fabric", path="build_moe_swizzle_task_placement")
    work_by_task = {}
    comps = tuple(task for task in projection.tasks if task.kind is SwizzleActionKind.COMP)
    tasks = {task.id: task for task in projection.tasks}
    values = {value.id: value for value in projection.values}

    swiglu_group_by_assignments = {}
    for task in projection.tasks:
        if task.kind is not SwizzleActionKind.SWIGLU:
            continue
        group_key = (
            task.expert_index,
            tuple(sorted(task.assignment_refs)),
        )
        prior = swiglu_group_by_assignments.get(group_key)
        if prior is not None and prior != task.pipeline_index:
            raise SchemaError(
                "exact SWIGLU assignment group maps to multiple placement groups",
                path="build_moe_swizzle_task_placement.projection.tasks",
            )
        swiglu_group_by_assignments[group_key] = task.pipeline_index

    def exact_assignment_group(task: object) -> int:
        if len(task.assignment_refs) == 1:
            return task.tile_index
        return swiglu_group_by_assignments.get(
            (task.expert_index, tuple(sorted(task.assignment_refs))),
            task.pipeline_index,
        )

    for task in projection.tasks:
        key = _task_work_key(task)
        if (
            task.rank == task.expert_index
            and task.kind in (SwizzleActionKind.COMP, SwizzleActionKind.SWIGLU)
        ):
            key = MoeSemanticWorkKey(
                key.domain, key.token_index, key.expert_index, key.tile_index,
                key.n_block, key.role, exact_assignment_group(task),
            )
        relay_send = (
            task.kind is SwizzleActionKind.SEND
            and task.work_role in ("moe_dispatch_gemm.transport", "moe_gemm_combine.transport")
            and len(task.read_value_refs) == 1
            and values[task.read_value_refs[0]].producer_task_ref is not None
            and tasks[values[task.read_value_refs[0]].producer_task_ref].kind
            is SwizzleActionKind.RECV
        )
        if relay_send:
            key = MoeSemanticWorkKey(
                "dispatch_expert_tile"
                if task.work_role == "moe_dispatch_gemm.transport"
                else "combine_source_token",
                None
                if task.work_role == "moe_dispatch_gemm.transport"
                else task.tile_index,
                task.expert_index, task.tile_index, None, task.work_role,
                task.pipeline_index,
            )
        needs_expert_tile = (
            task.work_role == "moe_gemm_combine.transport"
            and task.kind is SwizzleActionKind.SEND
            and not relay_send
        ) or (
            task.work_role == "moe_dispatch_gemm.transport"
            and (
                task.kind in (SwizzleActionKind.RECV, SwizzleActionKind.WAIT)
                or relay_send
            )
        )
        if needs_expert_tile:
            expected_roles = (
                ("gate", "up")
                if task.work_role == "moe_dispatch_gemm.transport"
                else ("down",)
            )
            assignment_groups = []
            for assignment_ref in task.assignment_refs:
                groups = {
                    (comp.expert_index, comp.tile_index, exact_assignment_group(comp))
                    for comp in comps
                    if comp.expert_index == task.expert_index
                    and comp.work_role in expected_roles
                    and assignment_ref in comp.assignment_refs
                }
                if len(groups) != 1:
                    raise SchemaError(
                        "transport assignment does not select one exact placement group",
                        path="build_moe_swizzle_task_placement.projection.tasks",
                    )
                assignment_groups.append(next(iter(groups)))
            if not assignment_groups or len(set(assignment_groups)) != 1:
                raise SchemaError("transport assignments do not select one exact expert tile", path="build_moe_swizzle_task_placement.projection.tasks")
            expert, tile, placement_group = assignment_groups[0]
            # M1 UNFUSED pipeline indices are transport-session provenance,
            # not grouped compute placement.  Grouped fused transport must
            # carry the exact M-block placement group end to end.
            if len(task.assignment_refs) > 1 and task.pipeline_index != placement_group:
                raise SchemaError(
                    "transport pipeline does not match its exact placement group",
                    path="build_moe_swizzle_task_placement.projection.tasks",
                )
            key = MoeSemanticWorkKey(
                key.domain, key.token_index, expert, tile,
                task.n_block if key.domain in ("combine_down_tile", "comp") else None,
                key.role,
                placement_group,
            )
        work_by_task[task.id] = key
    owner_map = {
        (owner.rank, owner.work_key): owner
        for owner in build_moe_work_owner_map(
            hardware_facts,
            tuple(MoeSemanticWorkItem(task.rank, work_by_task[task.id]) for task in projection.tasks),
        )
    }
    return tuple((task.id, owner_map[(task.rank, work_by_task[task.id])]) for task in projection.tasks)


def build_moe_swizzle_workload_placement(
    ir1: IR1,
    workload_projection: object,
    replacement_projection: object,
    hardware_facts: MoeHardwareFacts,
) -> tuple[MoeWorkloadActionOwner, ...]:
    """Place preserved actions by exact dependency affinity to replacement M-blocks."""

    workload_projection.validate("build_moe_swizzle_workload_placement.workload_projection")
    replacement_projection.validate("build_moe_swizzle_workload_placement.replacement_projection")
    if workload_projection.replacement_projection_id != replacement_projection.id:
        raise SchemaError("whole/replacement projection lineage is not exact", path="build_moe_swizzle_workload_placement")
    actions = {item.id: item for item in workload_projection.actions}
    replacement_tasks = {item.id: item for item in replacement_projection.tasks}
    replacement = dict(build_moe_swizzle_task_placement(
        ir1, replacement_projection, hardware_facts,
    ))
    owners = {}
    for ref, owner in replacement.items():
        task = replacement_tasks[ref]
        owners[ref] = MoeWorkloadActionOwner(
            ref, task.rank, owner.logical_core, owner.runtime_core_id,
            task.pipeline_index, (ref,),
        )
    consumers = {
        ref: tuple(item.id for item in workload_projection.actions if ref in item.deps)
        for ref in actions
    }
    remaining = {ref for ref, item in actions.items() if item.preserved}
    while remaining:
        progress = False
        for ref in sorted(remaining):
            action = actions[ref]
            witnesses = (
                consumers[ref]
                if action.kind == "preserved.dma_in"
                else action.deps
            )
            if not witnesses or any(item not in owners for item in witnesses):
                continue
            witness_cores = {(
                owners[item].logical_core,
                owners[item].runtime_core_id,
            ) for item in witnesses}
            if len(witness_cores) != 1:
                raise SchemaError(
                    "preserved action spans multiple M-block owners without LOCAL_COPY",
                    path="build_moe_swizzle_workload_placement.workload_projection",
                )
            logical, runtime = next(iter(witness_cores))
            group = min(owners[item].placement_group_index for item in witnesses)
            owner = MoeWorkloadActionOwner(
                ref, action.rank, logical, runtime, group,
                tuple(sorted(witnesses)),
            )
            owner.validate("build_moe_swizzle_workload_placement.owner")
            owners[ref] = owner
            remaining.remove(ref)
            progress = True
        if not progress:
            raise SchemaError(
                "preserved action lacks an exact placed dependency/consumer witness",
                path="build_moe_swizzle_workload_placement.workload_projection",
            )
    result = tuple(owners[item.id] for item in workload_projection.actions)
    for index, owner in enumerate(result):
        owner.validate(f"build_moe_swizzle_workload_placement.result[{index}]")
    return result


def _build_endpoint_widths_from_exact_dag(
    actions: dict[str, object],
    deps_by_ref: dict[str, tuple[str, ...]],
    endpoint_refs: set[str],
    owners: dict[str, object],
) -> tuple[MoeEndpointCoreWidth, ...]:
    if set(deps_by_ref) != set(actions) or not endpoint_refs.issubset(actions) or set(owners) != set(actions):
        raise SchemaError("endpoint DAG/action/owner coverage is not exact", path="moe_endpoint_widths")
    ancestors = {}

    def collect(ref: str) -> set[str]:
        if ref not in ancestors:
            result = set(deps_by_ref[ref])
            for dependency in deps_by_ref[ref]:
                if dependency not in actions:
                    raise SchemaError("endpoint DAG dependency is unknown", path="moe_endpoint_widths")
                result.update(collect(dependency))
            ancestors[ref] = result
        return ancestors[ref]

    for ref in actions:
        collect(ref)
    by_core = {}
    for ref in endpoint_refs:
        owner = owners[ref]
        by_core.setdefault((owner.logical_core, owner.runtime_core_id), []).append(ref)
    result = []
    for (logical, runtime), refs in sorted(
        by_core.items(),
        key=lambda item: (item[0][0].die_id, item[0][0].local_core_id),
    ):
        refs = tuple(sorted(refs))
        edges = {
            left: tuple(right for right in refs if left in ancestors[right])
            for left in refs
        }
        matched = {}

        def augment(left: str, seen: set[str]) -> bool:
            for right in edges[left]:
                if right in seen:
                    continue
                seen.add(right)
                if right not in matched or augment(matched[right], seen):
                    matched[right] = left
                    return True
            return False

        matching = sum(augment(left, set()) for left in refs)
        witness = MoeEndpointCoreWidth(logical, runtime, refs, len(refs) - matching)
        witness.validate("moe_endpoint_widths.result")
        result.append(witness)
    return tuple(result)


def measure_moe_swizzle_workload_endpoint_widths(
    workload_projection: object,
    replacement_projection: object,
    workload_placement: tuple[MoeWorkloadActionOwner, ...],
    *,
    capacity_per_core: int,
) -> tuple[MoeEndpointCoreWidth, ...]:
    """Compute exact endpoint-session antichain width in the whole overlay DAG."""

    workload_projection.validate("build_moe_swizzle_workload_endpoint_widths.workload_projection")
    replacement_projection.validate("build_moe_swizzle_workload_endpoint_widths.replacement_projection")
    validate_uint64(capacity_per_core, "build_moe_swizzle_workload_endpoint_widths.capacity_per_core")
    if capacity_per_core == 0:
        raise SchemaError("endpoint capacity must be positive", path="build_moe_swizzle_workload_endpoint_widths.capacity_per_core")
    actions = {item.id: item for item in workload_projection.actions}
    owners = {}
    for index, owner in enumerate(workload_placement):
        if type(owner) is not MoeWorkloadActionOwner:
            raise SchemaError("requires exact MoeWorkloadActionOwner", path=f"build_moe_swizzle_workload_endpoint_widths.workload_placement[{index}]")
        owner.validate(f"build_moe_swizzle_workload_endpoint_widths.workload_placement[{index}]")
        if owner.action_ref in owners:
            raise SchemaError("workload action owner is duplicated", path="build_moe_swizzle_workload_endpoint_widths.workload_placement")
        owners[owner.action_ref] = owner
    if set(owners) != set(actions):
        raise SchemaError("workload placement action coverage is not exact", path="build_moe_swizzle_workload_endpoint_widths.workload_placement")

    endpoint_refs = {
        ref
        for flow in replacement_projection.flows
        for ref in (flow.send_task_ref, flow.recv_task_ref)
    }
    result = _build_endpoint_widths_from_exact_dag(
        actions,
        {ref: tuple(action.deps) for ref, action in actions.items()},
        endpoint_refs,
        owners,
    )
    return result


def build_moe_swizzle_workload_endpoint_feasibility(
    workload_projection: object,
    replacement_projection: object,
    workload_placement: tuple[MoeWorkloadActionOwner, ...],
    *,
    capacity_per_core: int,
) -> MoeWorkloadEndpointFeasibility:
    """Return non-throwing whole-DAG endpoint truth for selection/admission."""

    widths = measure_moe_swizzle_workload_endpoint_widths(
        workload_projection, replacement_projection, workload_placement,
        capacity_per_core=capacity_per_core,
    )
    peak = max(item.max_inflight for item in widths)
    return MoeWorkloadEndpointFeasibility.create(
        workload_projection_id=workload_projection.id,
        replacement_projection_id=replacement_projection.id,
        placement_digest=canonical_digest(workload_placement),
        widths=widths,
        capacity_per_core=capacity_per_core,
        peak_width=peak,
        feasible=peak <= capacity_per_core,
    )


def build_moe_swizzle_whole_pair_feasibility(
    candidate_refs: tuple[str, str],
    workload_projection: object,
    replacement_projection: object,
    workload_placement: tuple[MoeWorkloadActionOwner, ...],
    workload_abi: object,
    *,
    capacity_per_core: int,
    dynamic_sram_capacity_bytes: int,
) -> MoeWholePairFeasibility:
    """Freeze joint selection truth from the exact tentative whole lowering."""

    workload_abi.validate("build_moe_swizzle_whole_pair_feasibility.workload_abi")
    validate_uint64(
        dynamic_sram_capacity_bytes,
        "build_moe_swizzle_whole_pair_feasibility.dynamic_sram_capacity_bytes",
    )
    if (
        workload_abi.source_workload_projection_id != workload_projection.id
        or workload_abi.source_replacement_projection_id != replacement_projection.id
    ):
        raise SchemaError(
            "whole-pair ABI lineage is not exact",
            path="build_moe_swizzle_whole_pair_feasibility.workload_abi",
        )
    endpoint = build_moe_swizzle_workload_endpoint_feasibility(
        workload_projection, replacement_projection, workload_placement,
        capacity_per_core=capacity_per_core,
    )
    dynamic_roots = tuple(
        item for item in workload_abi.roots
        if item.family in ("dispatch_operand", "combine_output")
    )
    keys = tuple(sorted(
        (item.runtime_core_id, item.family, item.slot) for item in dynamic_roots
    ))
    high_water = 0
    for runtime_core_id in sorted({item.runtime_core_id for item in dynamic_roots}):
        roots = tuple(item for item in dynamic_roots if item.runtime_core_id == runtime_core_id)
        for point in sorted({value for item in roots for value in (item.lifetime_start, item.lifetime_end_exclusive)}):
            high_water = max(high_water, sum(
                item.extent_bytes for item in roots
                if item.lifetime_start <= point < item.lifetime_end_exclusive
            ))
    feasible = endpoint.feasible and high_water <= dynamic_sram_capacity_bytes
    depth_by_key = {}
    for assignment in workload_projection.storage_slot_assignments:
        key = (assignment.runtime_core_id, assignment.family)
        depth_by_key[key] = max(depth_by_key.get(key, 0), assignment.slot + 1)
    color_depths = tuple(
        MoeStorageIntervalColorDepth(runtime, family, depth)
        for (runtime, family), depth in sorted(depth_by_key.items())
    )
    placement_by_ref = {
        item.action_ref: item for item in workload_placement
    }
    if len(placement_by_ref) != len(workload_placement):
        raise SchemaError(
            "whole placement duplicates action ownership",
            path="build_moe_swizzle_whole_pair_feasibility.workload_placement",
        )
    lifecycle = {
        runtime: [0, 0, 0]
        for runtime in sorted({
            item.runtime_core_id for item in workload_placement
        })
    }
    for root in workload_abi.roots:
        local_refs = tuple(
            ref for ref in root.action_refs
            if ref in placement_by_ref
            and placement_by_ref[ref].runtime_core_id == root.runtime_core_id
        )
        if not local_refs:
            raise SchemaError(
                "whole root lacks a same-core lifecycle owner",
                path="build_moe_swizzle_whole_pair_feasibility.workload_abi",
            )
        if root.allocate:
            lifecycle[root.runtime_core_id][0] += 1
        if root.free:
            lifecycle[root.runtime_core_id][2] += 1
    replacement_tasks = {
        item.id: item for item in replacement_projection.tasks
    }
    for action in workload_projection.actions:
        task = replacement_tasks.get(action.replacement_task_ref)
        if task is not None and task.kind is SwizzleActionKind.SWIGLU:
            lifecycle[placement_by_ref[action.id].runtime_core_id][1] += 1
    core_lifecycle_counts = tuple(
        MoeWholeCoreLifecycleCount(runtime, *counts)
        for runtime, counts in sorted(lifecycle.items())
    )
    return MoeWholePairFeasibility.create(
        candidate_refs=candidate_refs,
        placement=MoeWholePairPlacementFeasibility.create(
            workload_projection_id=workload_projection.id,
            replacement_projection_id=replacement_projection.id,
            placement_digest=canonical_digest(workload_placement),
            reason=MoeWholePairPlacementReason.ADMITTED,
            failure_path=None, failure_message=None, feasible=True,
        ),
        endpoint=endpoint,
        workload_abi_id=workload_abi.id, dynamic_root_keys=keys,
        dynamic_sram_high_water_bytes=high_water,
        dynamic_sram_capacity_bytes=dynamic_sram_capacity_bytes,
        storage_color_depths=color_depths,
        core_lifecycle_counts=core_lifecycle_counts,
        whole_physical_root_count=len(workload_abi.roots),
        whole_alloc_count=workload_abi.alloc_count,
        whole_free_count=workload_abi.free_count,
        feasible=feasible,
    )


def build_moe_swizzle_workload_endpoint_widths(
    workload_projection: object,
    replacement_projection: object,
    workload_placement: tuple[MoeWorkloadActionOwner, ...],
    *,
    capacity_per_core: int,
) -> tuple[MoeEndpointCoreWidth, ...]:
    witness = build_moe_swizzle_workload_endpoint_feasibility(
        workload_projection, replacement_projection, workload_placement,
        capacity_per_core=capacity_per_core,
    )
    for width in witness.widths:
        if width.max_inflight > capacity_per_core:
            raise SchemaError(
                f"whole-workload endpoint antichain {width.max_inflight} exceeds per-core capacity {capacity_per_core} on runtime core {width.runtime_core_id}",
                path="build_moe_swizzle_workload_endpoint_widths",
            )
    return witness.widths


def build_moe_action_dynamic_root_bindings(
    problem: MoeSwizzleProblem,
    pattern: FusionPattern,
    algorithm: SwizzleAlgorithm,
    actions: tuple[MoeActionWitness, ...],
) -> tuple[tuple[str, tuple[int, str, int]], ...]:
    """Bind each actual dynamic buffer user to its canonical physical root."""

    from .swizzle_moe import MoeActionWitness

    problem.validate("build_moe_action_dynamic_root_bindings.problem")
    if pattern not in (
        FusionPattern.MOE_DISPATCH_GEMM, FusionPattern.MOE_GEMM_COMBINE,
    ) or type(algorithm) is not SwizzleAlgorithm:
        raise SchemaError(
            "unsupported typed pattern/algorithm",
            path="build_moe_action_dynamic_root_bindings",
        )
    if type(actions) is not tuple or not actions:
        raise SchemaError(
            "actions must be a nonempty tuple",
            path="build_moe_action_dynamic_root_bindings.actions",
        )
    for index, action in enumerate(actions):
        if type(action) is not MoeActionWitness:
            raise SchemaError(
                "requires exact MoeActionWitness",
                path=f"build_moe_action_dynamic_root_bindings.actions[{index}]",
            )
        action.validate(f"build_moe_action_dynamic_root_bindings.actions[{index}]")

    expected_family = (
        "dispatch_operand"
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else "combine_output"
    )
    assignment_index = {
        item.id: item
        for item in problem.region.semantic_witness.traffic.assignments
    }
    action_index = {action.id: action for action in actions}

    def owns_dynamic_storage(action: MoeActionWitness) -> bool:
        remote = any(
            assignment_index[ref].source_rank
            != assignment_index[ref].expert_rank
            for ref in action.assignment_refs
        )
        if algorithm is SwizzleAlgorithm.UNFUSED:
            if (
                pattern is FusionPattern.MOE_DISPATCH_GEMM
                and action.kind is SwizzleActionKind.SWIGLU
            ):
                return True
            return remote and (
                action.kind in (
                    SwizzleActionKind.RECV,
                    SwizzleActionKind.COMP,
                )
                if pattern is FusionPattern.MOE_DISPATCH_GEMM
                else action.kind in (SwizzleActionKind.COMP, SwizzleActionKind.SEND)
            )
        if pattern is FusionPattern.MOE_DISPATCH_GEMM and action.kind is SwizzleActionKind.SWIGLU:
            return True
        if action.kind is SwizzleActionKind.COMP:
            return (
                pattern is FusionPattern.MOE_DISPATCH_GEMM
                or any(
                    assignment_index[ref].source_rank != action.rank
                    for ref in action.assignment_refs
                )
            )
        if pattern is FusionPattern.MOE_DISPATCH_GEMM:
            if action.kind is SwizzleActionKind.RECV:
                return True
            return action.kind is SwizzleActionKind.SEND and any(
                action_index[ref].kind is SwizzleActionKind.WAIT
                for ref in action.deps
            )
        if action.kind is SwizzleActionKind.SEND:
            return True
        if action.kind is SwizzleActionKind.RECV:
            return any(
                assignment_index[ref].source_rank != action.rank
                for ref in action.assignment_refs
            )
        return False

    owners = build_moe_candidate_action_owner_map(problem, actions)
    bindings = []
    for action in actions:
        owns = owns_dynamic_storage(action)
        if owns:
            if (
                action.buffer_family != expected_family
                or action.buffer_slot not in (0, 1)
                or (algorithm is SwizzleAlgorithm.UNFUSED and action.buffer_slot != 0)
            ):
                raise SchemaError(
                    "dynamic owner lacks its exact family/slot witness",
                    path="build_moe_action_dynamic_root_bindings.actions",
                )
            bindings.append((
                action.id,
                (
                    owners[action.id].runtime_core_id,
                    expected_family,
                    action.buffer_slot,
                ),
            ))
        elif (
            algorithm is SwizzleAlgorithm.UNFUSED
            and action.kind is not SwizzleActionKind.SWIGLU
            and (
                action.buffer_family is not None or action.buffer_slot is not None
            )
        ):
            raise SchemaError(
                "UNFUSED non-owner claims dynamic storage",
                path="build_moe_action_dynamic_root_bindings.actions",
            )
    if algorithm is not SwizzleAlgorithm.UNFUSED:
        if {action.buffer_family for action in actions} != {expected_family}:
            raise SchemaError(
                "action buffer family drifted",
                path="build_moe_action_dynamic_root_bindings.actions",
            )
        slots = {action.buffer_slot for action in actions}
        if slots not in ({0}, {0, 1}):
            raise SchemaError(
                "dynamic slots are not contiguous",
                path="build_moe_action_dynamic_root_bindings.actions",
            )
    if not bindings:
        raise SchemaError(
            "candidate has no actual dynamic buffer users",
            path="build_moe_action_dynamic_root_bindings.actions",
        )
    return tuple(sorted(bindings))


def build_moe_action_dynamic_root_keys(
    problem: MoeSwizzleProblem,
    pattern: FusionPattern,
    algorithm: SwizzleAlgorithm,
    actions: tuple[MoeActionWitness, ...],
) -> tuple[tuple[int, str, int], ...]:
    """Return planner-owned dynamic roots from actual typed buffer users."""

    return tuple(sorted({
        key for _, key in build_moe_action_dynamic_root_bindings(
            problem, pattern, algorithm, actions,
        )
    }))


def build_moe_candidate_dynamic_root_keys(
    problem: MoeSwizzleProblem,
    candidate: object,
) -> tuple[tuple[int, str, int], ...]:
    """Return per-region physical runtime-core/family/slot roots."""

    problem.validate("build_moe_candidate_dynamic_root_keys.problem")
    candidate.validate("build_moe_candidate_dynamic_root_keys.candidate")
    candidate.validate_against(problem, "build_moe_candidate_dynamic_root_keys.candidate")

    actions = tuple(
        action for program in candidate.rank_programs for action in program.actions
    )
    return build_moe_action_dynamic_root_keys(
        problem, candidate.pattern, candidate.algorithm, actions,
    )


def build_moe_candidate_core_lifecycle_floor(
    problem: MoeSwizzleProblem,
    candidate: object,
) -> tuple[MoeCandidateCoreLifecycleFloor, ...]:
    """Return a safe candidate-local ALLOC/BIND/FREE floor.

    The witness uses only typed candidate actions, assignment provenance, and
    the shared canonical owner map.  It deliberately excludes BORROWED
    boundary inputs and never derives a common term by subtracting a baseline.
    """

    problem.validate("build_moe_candidate_core_lifecycle_floor.problem")
    candidate.validate("build_moe_candidate_core_lifecycle_floor.candidate")
    candidate.validate_against(
        problem, "build_moe_candidate_core_lifecycle_floor.candidate",
    )
    actions = tuple(
        action for program in candidate.rank_programs
        for action in program.actions
    )
    owners = build_moe_candidate_action_owner_map(problem, actions)
    assignments = {
        item.id: item
        for item in problem.region.semantic_witness.traffic.assignments
    }
    roots: set[tuple[int, str, int]] = set(
        build_moe_candidate_dynamic_root_keys(problem, candidate)
    )
    terminal_roots: set[tuple[int, str, int]] = set()
    binds: dict[int, int] = {}

    if candidate.pattern is FusionPattern.MOE_DISPATCH_GEMM:
        for action in actions:
            runtime = owners[action.id].runtime_core_id
            if action.kind is SwizzleActionKind.COMP:
                if action.work_role not in ("gate", "up"):
                    raise SchemaError(
                        "Dispatch COMP role cannot define lifecycle floor",
                        path="build_moe_candidate_core_lifecycle_floor.candidate",
                    )
                roots.add((
                    runtime, f"state_stage.{action.work_role}",
                    0 if action.buffer_slot is None else action.buffer_slot,
                ))
            elif action.kind is SwizzleActionKind.SWIGLU:
                # Whole storage coloring may require slot 1, but every active
                # SwiGLU core necessarily owns at least one output root.
                roots.add((runtime, "swiglu_output", 0))
                binds[runtime] = binds.get(runtime, 0) + 1
    elif candidate.pattern is FusionPattern.MOE_GEMM_COMBINE:
        for action in actions:
            runtime = owners[action.id].runtime_core_id
            if action.kind is SwizzleActionKind.COMP:
                if action.work_role != "down":
                    raise SchemaError(
                        "Combine COMP role cannot define lifecycle floor",
                        path="build_moe_candidate_core_lifecycle_floor.candidate",
                    )
                roots.add((
                    runtime, "state_stage.down",
                    0 if action.buffer_slot is None else action.buffer_slot,
                ))
                local_terminal = any(
                    assignments[ref].source_rank
                    == assignments[ref].expert_rank
                    == action.rank
                    for ref in action.assignment_refs
                )
                if local_terminal:
                    terminal_roots.add((runtime, "terminal_combined", 0))
            elif action.kind is SwizzleActionKind.RECV and any(
                assignments[ref].source_rank == action.rank
                for ref in action.assignment_refs
            ):
                terminal_roots.add((runtime, "terminal_combined", 0))
        roots.update(terminal_roots)
    else:
        raise SchemaError(
            "unsupported candidate lifecycle-floor pattern",
            path="build_moe_candidate_core_lifecycle_floor.candidate.pattern",
        )

    by_core: dict[int, list[int]] = {}
    for runtime, family, slot in sorted(roots):
        del slot
        counts = by_core.setdefault(runtime, [0, 0, 0])
        counts[0] += 1
        if family != "terminal_combined":
            counts[2] += 1
    for runtime, count in binds.items():
        by_core.setdefault(runtime, [0, 0, 0])[1] += count
    result = tuple(
        MoeCandidateCoreLifecycleFloor(candidate.id, runtime, *counts)
        for runtime, counts in sorted(by_core.items())
    )
    for index, item in enumerate(result):
        item.validate(f"build_moe_candidate_core_lifecycle_floor.result[{index}]")
    return result


def build_moe_candidate_core_fixed_lifecycle_floor(
    problem: MoeSwizzleProblem,
    candidate: object,
) -> tuple[MoeCandidateCoreLifecycleFloor, ...]:
    """Return only the non-dynamic part of the candidate lifecycle floor.

    Joint cost code already carries exact dynamic root counts separately.  This
    view prevents charging those roots twice while retaining the full-floor API
    used to cross-check a materialized whole workload.
    """

    total = {
        item.runtime_core_id: item
        for item in build_moe_candidate_core_lifecycle_floor(problem, candidate)
    }
    dynamic_counts: dict[int, int] = {}
    for runtime_core_id, _family, _slot in build_moe_candidate_dynamic_root_keys(
        problem, candidate,
    ):
        dynamic_counts[runtime_core_id] = (
            dynamic_counts.get(runtime_core_id, 0) + 1
        )
    result = []
    for runtime_core_id, item in sorted(total.items()):
        dynamic = dynamic_counts.get(runtime_core_id, 0)
        if dynamic > item.alloc_count or dynamic > item.free_count:
            raise SchemaError(
                "dynamic lifecycle exceeds total candidate floor",
                path="build_moe_candidate_core_fixed_lifecycle_floor",
            )
        counts = (
            item.alloc_count - dynamic,
            item.bind_count,
            item.free_count - dynamic,
        )
        if any(counts):
            result.append(MoeCandidateCoreLifecycleFloor(
                candidate.id, runtime_core_id, *counts,
            ))
    frozen = tuple(result)
    for index, item in enumerate(frozen):
        item.validate(
            f"build_moe_candidate_core_fixed_lifecycle_floor.result[{index}]"
        )
    return frozen

def build_moe_projection_dynamic_root_keys(
    ir1: IR1,
    projection: object,
    hardware_facts: MoeHardwareFacts,
) -> tuple[tuple[int, str, int], ...]:
    """Return whole-overlay physical roots from typed task buffer uses."""

    ir1.validate("build_moe_projection_dynamic_root_keys.ir1")
    projection.validate("build_moe_projection_dynamic_root_keys.projection")
    placement = dict(build_moe_swizzle_task_placement(ir1, projection, hardware_facts))
    keys = set()
    for task in projection.tasks:
        for use in task.buffer_uses:
            family = use.buffer_ref.rsplit(".", 1)[-1]
            if family not in ("dispatch_operand", "combine_output"):
                raise SchemaError("projection buffer lacks a typed dynamic family", path="build_moe_projection_dynamic_root_keys.projection")
            keys.add((placement[task.id].runtime_core_id, family, use.slot))
    return tuple(sorted(keys))


__all__ = [
    "MoeSemanticWorkItem", "MoeSemanticWorkKey", "MoeWorkOwner",
    "MoeEndpointCoreWidth", "MoeWorkloadActionOwner",
    "MoeWorkloadEndpointFeasibility", "MoeWholePairFeasibility",
    "MoeWholePairPlacementFeasibility", "MoeWholePairPlacementReason",
    "build_moe_candidate_action_owner_map",
    "build_moe_action_dynamic_root_bindings", "build_moe_action_dynamic_root_keys",
    "build_moe_candidate_dynamic_root_keys", "build_moe_candidate_core_lifecycle_floor",
    "build_moe_candidate_core_fixed_lifecycle_floor",
    "build_moe_projection_dynamic_root_keys",
    "build_moe_swizzle_task_placement", "build_moe_swizzle_workload_placement",
    "measure_moe_swizzle_workload_endpoint_widths",
    "build_moe_swizzle_workload_endpoint_feasibility",
    "build_moe_swizzle_whole_pair_feasibility",
    "build_moe_swizzle_workload_endpoint_widths",
    "build_moe_work_owner_map",
]
