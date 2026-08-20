"""Profile-complete provenance wrapper for the N3 placement output."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import PlacementStrategy
from .ir0 import (
    DeviceMesh,
    IR0,
    InstanceProfileBinding,
    LogicalInstance,
    LogicalNode,
    TrainStructure,
)
from .ir1 import IR1, PhysicalGroup, PhysicalInstance, PhysicalNode
from .logical import ExpandedIR0Bundle, ExpandedProfileIR0, ProfileEntry
from .placement import PlacementContext
from .stage4_pd import Stage4PdMode, Stage4PdPlan


PLACED_IR1_BUNDLE_SCHEMA_VERSION = "wafer_frontend.placed_ir1_bundle/v1alpha5"
STAGE4_PLACED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.stage4_placed_ir1/v1alpha2"
)
TRAIN_PLACED_IR1_SCHEMA_VERSION = "wafer_frontend.train_placed_ir1/v1alpha1"


def _train_logical_signature(graph: IR1) -> tuple[object, ...]:
    """Quotient replica-local physical ids back to one exact logical graph."""

    node_origins = {node.id: node.origin_node_id for node in graph.nodes}
    return (
        graph.source_ir0_id,
        graph.profile,
        tuple(
            (
                node.origin_node_id,
                node.instance_id,
                node.kind,
                node.phase,
                node.stage,
                node.mesh_ref,
                node.inputs,
                node.outputs,
                node.workload,
                node.math,
                node.effects,
                node.impl_ref,
            )
            for node in graph.nodes
        ),
        tuple(
            (
                value.id,
                value.shape,
                value.dtype,
                value.logical_layout,
                value.sharding,
                None if value.producer is None else node_origins[value.producer],
                tuple(node_origins[item] for item in value.consumers),
                value.alias_set,
            )
            for value in graph.values
        ),
        tuple(
            (
                edge.kind,
                node_origins[edge.source_node],
                node_origins[edge.destination_node],
                edge.value_id,
            )
            for edge in graph.edges
        ),
        tuple(
            (
                tuple(node_origins[item] for item in candidate.members),
                candidate.boundary_inputs,
                candidate.boundary_outputs,
                candidate.semantic_contract,
                candidate.impl,
                candidate.origin,
            )
            for candidate in graph.fusion_candidates
        ),
        tuple(
            (
                node_origins[access.node_ref],
                access.state_ref,
                access.mode,
                access.rank,
                access.read_offset,
                access.read_shape,
                access.write_offset,
                access.write_shape,
            )
            for access in graph.state_accesses
        ),
        graph.instance_profiles,
        tuple(
            (node_origins[binding.node_ref], binding.profile)
            for binding in graph.node_profiles
        ),
        graph.pd_plan_id,
    )


@dataclass(frozen=True, slots=True)
class TrainPlacedReplica:
    id: str
    replica_index: int
    graph: IR1

    @classmethod
    def create(cls, *, replica_index: int, graph: IR1) -> "TrainPlacedReplica":
        semantic_key = {"replica_index": replica_index, "graph_id": graph.id}
        return cls(
            id=stable_artifact_id(
                "train_placed_replica",
                semantic_key,
                schema_version=TRAIN_PLACED_IR1_SCHEMA_VERSION,
            ),
            replica_index=replica_index,
            graph=graph,
        )

    def validate(self, path: str) -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "placement":
            raise SchemaError("must be produced by placement", path=f"{path}.graph.producer_pass")
        if len(self.graph.groups) != 1:
            raise SchemaError("replica IR1 must own exactly one TP group", path=f"{path}.graph.groups")
        group = self.graph.groups[0]
        if not group.id.endswith(f"__dp{self.replica_index}"):
            raise SchemaError("group id must carry canonical DP lineage", path=f"{path}.graph.groups[0].id")
        if any(node.execution_group_ref != group.id for node in self.graph.nodes):
            raise SchemaError("every node must execute in its replica TP group", path=f"{path}.graph.nodes")
        if any(
            not node.id.endswith(f"__dp{self.replica_index}")
            for node in self.graph.nodes
        ):
            raise SchemaError(
                "physical node ids must carry canonical DP lineage",
                path=f"{path}.graph.nodes",
            )
        if self.graph.persistent_state_manifest is None:
            raise SchemaError("train replica requires parameter backing", path=f"{path}.graph.persistent_state_manifest")
        expected_id = stable_artifact_id(
            "train_placed_replica",
            {"replica_index": self.replica_index, "graph_id": self.graph.id},
            schema_version=TRAIN_PLACED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable replica id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class TrainPlacedIR1:
    schema_version: str
    producer_pass: str
    id: str
    source_ir0_id: str
    placement_context_id: str
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainPlacedReplica, ...]

    @classmethod
    def create(
        cls,
        *,
        source: IR0,
        placement_context: PlacementContext,
        replicas: tuple[TrainPlacedReplica, ...],
    ) -> "TrainPlacedIR1":
        if source.train is None:
            raise SchemaError("TRAIN source requires TrainStructure", path="source.train")
        semantic_key = {
            "source_ir0_id": source.id,
            "placement_context_id": placement_context.id,
            "train_structure": source.train,
            "dp_degree": source.instances[0].parallel.dp,
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_PLACED_IR1_SCHEMA_VERSION,
            producer_pass="train_placement",
            id=stable_artifact_id(
                "train_placed_ir1",
                semantic_key,
                schema_version=TRAIN_PLACED_IR1_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_ir0_id": self.source_ir0_id,
            "placement_context_id": self.placement_context_id,
            "train_structure": self.train_structure,
            "dp_degree": self.dp_degree,
            "replicas": self.replicas,
        }

    def validate(self, path: str = "train_placed_ir1") -> None:
        if self.schema_version != TRAIN_PLACED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_placement":
            raise SchemaError("must be 'train_placement'", path=f"{path}.producer_pass")
        validate_nonempty(self.source_ir0_id, f"{path}.source_ir0_id")
        validate_nonempty(self.placement_context_id, f"{path}.placement_context_id")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError("must contain exactly dp_degree replicas", path=f"{path}.replicas")
        if tuple(replica.replica_index for replica in self.replicas) != tuple(range(self.dp_degree)):
            raise SchemaError("replica_index must be canonical 0..dp-1", path=f"{path}.replicas")

        used_dies: set[int] = set()
        graph_ids: set[str] = set()
        node_ids: set[str] = set()
        binding_ids: set[str] = set()
        reference: IR1 | None = None
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainPlacedReplica:
                raise SchemaError("must be a TrainPlacedReplica", path=replica_path)
            replica.validate(replica_path)
            graph = replica.graph
            if graph.source_ir0_id != self.source_ir0_id:
                raise SchemaError("must preserve source IR0", path=f"{replica_path}.graph.source_ir0_id")
            if graph.id in graph_ids:
                raise SchemaError("replica IR1 artifacts must be distinct", path=f"{replica_path}.graph.id")
            graph_ids.add(graph.id)
            local_node_ids = {node.id for node in graph.nodes}
            if node_ids.intersection(local_node_ids):
                raise SchemaError(
                    "physical node ids must be replica-distinct",
                    path=f"{replica_path}.graph.nodes",
                )
            node_ids.update(local_node_ids)
            group = graph.groups[0]
            replica_dies = {placement.die_id for placement in group.placements}
            if used_dies.intersection(replica_dies):
                raise SchemaError("DP replica TP groups must use disjoint dies", path=f"{replica_path}.graph.groups[0].placements")
            used_dies.update(replica_dies)
            manifest = graph.persistent_state_manifest
            assert manifest is not None
            local_binding_ids = {binding.id for binding in manifest.bindings}
            if binding_ids.intersection(local_binding_ids):
                raise SchemaError("physical state binding ids must be replica-distinct", path=f"{replica_path}.graph.persistent_state_manifest.bindings")
            binding_ids.update(local_binding_ids)
            if {binding.die_id for binding in manifest.bindings} - replica_dies:
                raise SchemaError("parameter backing must reside on the replica TP dies", path=f"{replica_path}.graph.persistent_state_manifest.bindings")
            if reference is None:
                reference = graph
                continue
            reference_manifest = reference.persistent_state_manifest
            assert reference_manifest is not None
            if (
                _train_logical_signature(graph)
                != _train_logical_signature(reference)
                or manifest.declarations != reference_manifest.declarations
            ):
                raise SchemaError("replicas must preserve one exact logical graph/state template", path=replica_path)
        expected_id = stable_artifact_id(
            "train_placed_ir1",
            self._semantic_key(),
            schema_version=TRAIN_PLACED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


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


@dataclass(frozen=True, slots=True)
class PlacedProfileIR1:
    id: str
    source_expanded_entry_id: str
    profile_id: str
    weight: float
    graph: IR1

    @classmethod
    def create(
        cls,
        *,
        source_expanded_entry_id: str,
        weight: float,
        graph: IR1,
    ) -> "PlacedProfileIR1":
        profile_id = graph.profile.stable_id()
        semantic_key = {
            "source_expanded_entry_id": source_expanded_entry_id,
            "profile_id": profile_id,
            "weight": weight,
            "graph_id": graph.id,
        }
        return cls(
            id=stable_artifact_id(
                "placed_profile_ir1",
                semantic_key,
                schema_version=PLACED_IR1_BUNDLE_SCHEMA_VERSION,
            ),
            source_expanded_entry_id=source_expanded_entry_id,
            profile_id=profile_id,
            weight=weight,
            graph=graph,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_expanded_entry_id": self.source_expanded_entry_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
        }

    def validate(self, path: str = "placed_entry") -> None:
        validate_nonempty(
            self.source_expanded_entry_id,
            f"{path}.source_expanded_entry_id",
        )
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "placement":
            raise SchemaError(
                "must be produced by placement",
                path=f"{path}.graph.producer_pass",
            )
        if self.graph.fused_op_skeletons:
            raise SchemaError(
                "must be empty before fusion_partition",
                path=f"{path}.graph.fused_op_skeletons",
            )
        if self.graph.cross_routes:
            raise SchemaError(
                "must be empty before inter_die_plan",
                path=f"{path}.graph.cross_routes",
            )
        expected_profile_id = self.graph.profile.stable_id()
        if self.profile_id != expected_profile_id:
            raise SchemaError(
                f"must equal graph profile id {expected_profile_id!r}",
                path=f"{path}.profile_id",
            )
        if (
            type(self.weight) is not float
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float",
                path=f"{path}.weight",
            )
        expected_id = stable_artifact_id(
            "placed_profile_ir1",
            self._semantic_key(),
            schema_version=PLACED_IR1_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )


def _expected_groups(
    logical_instances: tuple[LogicalInstance, ...],
) -> tuple[tuple[LogicalInstance, DeviceMesh], ...]:
    return tuple(
        (instance, mesh)
        for instance in logical_instances
        for mesh in instance.meshes
    )


def _canonical_die_region(groups: tuple[PhysicalGroup, ...]) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for placement in group.placements:
            if placement.die_id not in seen:
                seen.add(placement.die_id)
                result.append(placement.die_id)
    return tuple(result)


def _validate_group_against_mesh(
    group: PhysicalGroup,
    logical_instance: LogicalInstance,
    mesh: DeviceMesh,
    *,
    path: str,
) -> None:
    if group.instance_id != logical_instance.id:
        raise SchemaError(
            "must preserve logical instance ownership",
            path=f"{path}.instance_id",
        )
    if group.mesh_ref != mesh.id:
        raise SchemaError("must preserve logical mesh id", path=f"{path}.mesh_ref")
    if len(mesh.axes) != 1:
        raise SchemaError(
            "N3 Dense placement requires a one-axis logical mesh",
            path=f"{path}.logical_shape",
        )
    if group.axis is not mesh.axes[0].name:
        raise SchemaError("must preserve logical mesh axis", path=f"{path}.axis")
    expected_shape = tuple(axis.size for axis in mesh.axes)
    if group.logical_shape != expected_shape:
        raise SchemaError(
            "must equal logical mesh shape",
            path=f"{path}.logical_shape",
        )


def _validate_instance_against_logical(
    physical: PhysicalInstance,
    logical: LogicalInstance,
    logical_graph: IR0,
    owned_groups: tuple[PhysicalGroup, ...],
    *,
    path: str,
) -> None:
    if physical.id != logical.id or physical.origin_instance_id != logical.id:
        raise SchemaError(
            "physical id and origin must preserve the logical instance id",
            path=f"{path}.origin_instance_id",
        )
    if physical.role is not logical.role:
        raise SchemaError("must preserve logical role", path=f"{path}.role")
    expected_group_ids = tuple(group.id for group in owned_groups)
    if physical.group_ids != expected_group_ids:
        raise SchemaError(
            "must own exactly the logical mesh groups in source order",
            path=f"{path}.group_ids",
        )
    expected_node_ids = tuple(
        node.id for node in logical_graph.nodes if node.instance_id == logical.id
    )
    if physical.node_ids != expected_node_ids:
        raise SchemaError(
            "must own exactly the logical nodes in source order",
            path=f"{path}.node_ids",
        )
    expected_die_region = _canonical_die_region(owned_groups)
    if physical.die_region != expected_die_region:
        raise SchemaError(
            "must be the canonical first-seen union of owned group placements",
            path=f"{path}.die_region",
        )


_PRESERVED_NODE_FIELDS = (
    "instance_id",
    "kind",
    "phase",
    "stage",
    "mesh_ref",
    "inputs",
    "outputs",
    "workload",
    "math",
    "effects",
    "impl_ref",
)


def _validate_node_against_logical(
    physical: PhysicalNode,
    logical: LogicalNode,
    group_by_owner: dict[tuple[str, str], PhysicalGroup],
    *,
    path: str,
) -> None:
    logical_id = logical.id
    if physical.id != logical_id or physical.origin_node_id != logical_id:
        raise SchemaError(
            "physical id and origin must preserve the logical node id",
            path=f"{path}.origin_node_id",
        )
    for field_name in _PRESERVED_NODE_FIELDS:
        if getattr(physical, field_name) != getattr(logical, field_name):
            raise SchemaError(
                "must exactly preserve the logical node field",
                path=f"{path}.{field_name}",
            )
    owner = (physical.instance_id, physical.mesh_ref)
    expected_group = group_by_owner.get(owner)
    if expected_group is None:
        raise SchemaError(
            "logical node has no physical execution group",
            path=f"{path}.execution_group_ref",
        )
    if physical.execution_group_ref != expected_group.id:
        raise SchemaError(
            "must reference the group for the logical instance and mesh",
            path=f"{path}.execution_group_ref",
        )


def _validate_ir1_against_source(
    physical: IR1,
    source: ExpandedProfileIR0,
    context: PlacementContext,
    *,
    path: str,
) -> None:
    logical = source.graph
    if physical.source_ir0_id != logical.id:
        raise SchemaError(
            "must equal the source expanded IR0 graph id",
            path=f"{path}.source_ir0_id",
        )
    if physical.profile != logical.profile:
        raise SchemaError("must preserve the source profile", path=f"{path}.profile")
    if physical.fabric != context.fabric:
        raise SchemaError(
            "must equal the placement context fabric",
            path=f"{path}.fabric",
        )
    for field_name in ("values", "edges", "fusion_candidates"):
        if getattr(physical, field_name) != getattr(logical, field_name):
            raise SchemaError(
                "must exactly preserve the source IR0 field",
                path=f"{path}.{field_name}",
            )

    if physical.state_accesses != logical.state_accesses:
        raise SchemaError(
            "must exactly preserve logical state accesses",
            path=f"{path}.state_accesses",
        )
    if not logical.persistent_states:
        if physical.persistent_state_manifest is not None:
            raise SchemaError(
                "must be null when the logical graph has no persistent state",
                path=f"{path}.persistent_state_manifest",
            )
    else:
        manifest = physical.persistent_state_manifest
        if manifest is None:
            raise SchemaError(
                "logical persistent state requires physical HBM backing",
                path=f"{path}.persistent_state_manifest",
            )
        if manifest.declarations != logical.persistent_states:
            raise SchemaError(
                "must exactly preserve logical persistent state declarations",
                path=f"{path}.persistent_state_manifest.declarations",
            )
        if manifest.address_spaces != context.hbm_address_spaces:
            raise SchemaError(
                "must exactly preserve placement-context HBM address spaces",
                path=f"{path}.persistent_state_manifest.address_spaces",
            )

    expected_group_sources = _expected_groups(logical.instances)
    if len(physical.groups) != len(expected_group_sources):
        raise SchemaError(
            "must contain exactly one physical group per logical mesh",
            path=f"{path}.groups",
        )
    for index, (group, (logical_instance, mesh)) in enumerate(
        zip(physical.groups, expected_group_sources)
    ):
        _validate_group_against_mesh(
            group,
            logical_instance,
            mesh,
            path=f"{path}.groups[{index}]",
        )
    group_by_owner = {
        (group.instance_id, group.mesh_ref): group for group in physical.groups
    }

    if len(physical.instances) != len(logical.instances):
        raise SchemaError(
            "must contain exactly one physical instance per logical instance",
            path=f"{path}.instances",
        )
    for index, (physical_instance, logical_instance) in enumerate(
        zip(physical.instances, logical.instances)
    ):
        owned_groups = tuple(
            group
            for group in physical.groups
            if group.instance_id == logical_instance.id
        )
        _validate_instance_against_logical(
            physical_instance,
            logical_instance,
            logical,
            owned_groups,
            path=f"{path}.instances[{index}]",
        )

    if len(physical.nodes) != len(logical.nodes):
        raise SchemaError(
            "must contain exactly one physical node per logical node",
            path=f"{path}.nodes",
        )
    for index, (physical_node, logical_node) in enumerate(
        zip(physical.nodes, logical.nodes)
    ):
        _validate_node_against_logical(
            physical_node,
            logical_node,
            group_by_owner,
            path=f"{path}.nodes[{index}]",
        )

    if context.placement.strategy is PlacementStrategy.EXPLICIT:
        requested = {
            (item.instance_id, item.mesh_ref): item.die_ids
            for item in context.placement.groups
        }
        actual = {
            (group.instance_id, group.mesh_ref): tuple(
                placement.die_id for placement in group.placements
            )
            for group in physical.groups
        }
        if actual != requested:
            raise SchemaError(
                "physical rank placement must exactly match explicit placement",
                path=f"{path}.groups",
            )


@dataclass(frozen=True, slots=True)
class PlacedIR1Bundle:
    schema_version: str
    producer_pass: str
    id: str
    source_expanded_bundle_id: str
    placement_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[PlacedProfileIR1, ...]

    @classmethod
    def create(
        cls,
        *,
        source_expanded_bundle: ExpandedIR0Bundle,
        placement_context: PlacementContext,
        entries: tuple[PlacedProfileIR1, ...],
    ) -> "PlacedIR1Bundle":
        source_expanded_bundle.validate("source_expanded_bundle")
        placement_context.validate("placement_context")
        semantic_key = {
            "source_expanded_bundle_id": source_expanded_bundle.id,
            "placement_context_id": placement_context.id,
            "source_profiles": source_expanded_bundle.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=PLACED_IR1_BUNDLE_SCHEMA_VERSION,
            producer_pass="placement",
            id=stable_artifact_id(
                "placed_ir1_bundle",
                semantic_key,
                schema_version=PLACED_IR1_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_expanded_bundle_id": self.source_expanded_bundle_id,
            "placement_context_id": self.placement_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "placed_ir1_bundle") -> None:
        if self.schema_version != PLACED_IR1_BUNDLE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "placement":
            raise SchemaError("must be 'placement'", path=f"{path}.producer_pass")
        validate_nonempty(
            self.source_expanded_bundle_id,
            f"{path}.source_expanded_bundle_id",
        )
        validate_nonempty(self.placement_context_id, f"{path}.placement_context_id")
        _validate_profile_manifest(
            self.source_profiles,
            path=f"{path}.source_profiles",
        )
        if type(self.entries) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.entries")
        if len(self.entries) != len(self.source_profiles):
            raise SchemaError(
                "must contain exactly one IR1 graph per source profile",
                path=f"{path}.entries",
            )
        graph_ids: set[str] = set()
        source_entry_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        for index, (profile, entry) in enumerate(
            zip(self.source_profiles, self.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not PlacedProfileIR1:
                raise SchemaError("must be a PlacedProfileIR1", path=entry_path)
            entry.validate(entry_path)
            if entry.profile_id != profile.profile_id:
                raise SchemaError(
                    "must match the corresponding source profile",
                    path=f"{entry_path}.profile_id",
                )
            if entry.graph.profile != profile.key:
                raise SchemaError(
                    "graph profile must match the corresponding source profile",
                    path=f"{entry_path}.graph.profile",
                )
            if entry.weight != profile.weight:
                raise SchemaError(
                    "must match the corresponding source profile weight",
                    path=f"{entry_path}.weight",
                )
            if entry.source_expanded_entry_id in source_entry_ids:
                raise SchemaError(
                    "must reference a unique expanded entry",
                    path=f"{entry_path}.source_expanded_entry_id",
                )
            source_entry_ids.add(entry.source_expanded_entry_id)
            if entry.graph.id in graph_ids:
                raise SchemaError(
                    "each profile must own an independent IR1 graph",
                    path=f"{entry_path}.graph.id",
                )
            graph_ids.add(entry.graph.id)
            if reference_fabric is None:
                reference_fabric = entry.graph.fabric
                reference_groups = entry.graph.groups
            else:
                if entry.graph.fabric != reference_fabric:
                    raise SchemaError(
                        "all profile IR1 graphs must use the same fabric",
                        path=f"{entry_path}.graph.fabric",
                    )
                if entry.graph.groups != reference_groups:
                    raise SchemaError(
                        "all profile IR1 graphs must use identical groups, placements, and embeddings",
                        path=f"{entry_path}.graph.groups",
                    )
        expected_id = stable_artifact_id(
            "placed_ir1_bundle",
            self._semantic_key(),
            schema_version=PLACED_IR1_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: ExpandedIR0Bundle,
        context: PlacementContext,
        path: str = "placed_ir1_bundle",
    ) -> None:
        source.validate("source_expanded_bundle")
        context.validate("placement_context")
        self.validate(path)
        if self.source_expanded_bundle_id != source.id:
            raise SchemaError(
                "does not match the supplied expanded bundle",
                path=f"{path}.source_expanded_bundle_id",
            )
        if self.placement_context_id != context.id:
            raise SchemaError(
                "does not match the supplied placement context",
                path=f"{path}.placement_context_id",
            )
        if self.source_profiles != source.source_profiles:
            raise SchemaError(
                "must exactly preserve the expanded bundle profile manifest",
                path=f"{path}.source_profiles",
            )
        if len(self.entries) != len(source.entries):
            raise SchemaError(
                "must cover every expanded entry exactly once",
                path=f"{path}.entries",
            )
        for index, (entry, source_entry) in enumerate(
            zip(self.entries, source.entries)
        ):
            entry_path = f"{path}.entries[{index}]"
            if entry.source_expanded_entry_id != source_entry.id:
                raise SchemaError(
                    "must match the corresponding expanded entry",
                    path=f"{entry_path}.source_expanded_entry_id",
                )
            if entry.profile_id != source_entry.profile_id:
                raise SchemaError(
                    "must preserve the expanded entry profile",
                    path=f"{entry_path}.profile_id",
                )
            if entry.weight != source_entry.weight:
                raise SchemaError(
                    "must preserve the expanded entry weight",
                    path=f"{entry_path}.weight",
                )
            _validate_ir1_against_source(
                entry.graph,
                source_entry,
                context,
                path=f"{entry_path}.graph",
            )


@dataclass(frozen=True, slots=True)
class Stage4PlacedIR1:
    """Exact single-graph placement carrier for Stage 4 PD."""

    schema_version: str
    producer_pass: str
    id: str
    source_ir0_id: str
    placement_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1

    @classmethod
    def create(
        cls,
        *,
        source: IR0,
        context: PlacementContext,
        pd_plan: Stage4PdPlan,
        graph: IR1,
    ) -> "Stage4PlacedIR1":
        semantic_key = {
            "source_ir0_id": source.id,
            "placement_context_id": context.id,
            "pd_plan": pd_plan,
            "graph_id": graph.id,
        }
        return cls(
            schema_version=STAGE4_PLACED_IR1_SCHEMA_VERSION,
            producer_pass="placement",
            id=stable_artifact_id(
                "stage4_placed_ir1",
                semantic_key,
                schema_version=STAGE4_PLACED_IR1_SCHEMA_VERSION,
            ),
            source_ir0_id=source.id,
            placement_context_id=context.id,
            pd_plan=pd_plan,
            graph=graph,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_ir0_id": self.source_ir0_id,
            "placement_context_id": self.placement_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
        }

    def validate(self, path: str = "stage4_placed_ir1") -> None:
        if self.schema_version != STAGE4_PLACED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "placement":
            raise SchemaError("must be 'placement'", path=f"{path}.producer_pass")
        validate_nonempty(self.source_ir0_id, f"{path}.source_ir0_id")
        validate_nonempty(self.placement_context_id, f"{path}.placement_context_id")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError("must be a Stage4PdPlan", path=f"{path}.pd_plan")
        self.pd_plan.validate(f"{path}.pd_plan")
        if (
            self.pd_plan.mode is Stage4PdMode.FUSED
            and (self.pd_plan.prefill_tp != 1 or self.pd_plan.decode_tp != 1)
        ):
            raise SchemaError(
                "fused Stage 4 carrier supports TP1 only",
                path=f"{path}.pd_plan.mode",
            )
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "placement":
            raise SchemaError("must embed placement output", path=f"{path}.graph.producer_pass")
        if self.graph.source_ir0_id != self.source_ir0_id:
            raise SchemaError("must equal embedded graph source", path=f"{path}.source_ir0_id")
        if self.graph.pd_plan_id != self.pd_plan.id:
            raise SchemaError("graph must reference the embedded PD plan", path=f"{path}.graph.pd_plan_id")
        if self.graph.fused_op_skeletons:
            raise SchemaError("must be empty before fusion_partition", path=f"{path}.graph.fused_op_skeletons")
        expected_profiles = tuple(
            sorted(
                (
                    InstanceProfileBinding(
                        self.pd_plan.prefill_instance_ref,
                        self.pd_plan.prefill_profile.key,
                    ),
                    InstanceProfileBinding(
                        self.pd_plan.decode_instance_ref,
                        self.pd_plan.decode_profile.key,
                    ),
                ),
                key=lambda item: (
                    item.instance_ref,
                    item.profile.stable_id(),
                ),
            )
        )
        if self.graph.instance_profiles != expected_profiles:
            raise SchemaError("must exactly preserve PD instance profiles", path=f"{path}.graph.instance_profiles")
        groups = {group.id: group for group in self.graph.groups}
        actual_pairs = tuple(
            (
                groups[route.source_group_ref].instance_id,
                route.source_rank,
                groups[route.destination_group_ref].instance_id,
                route.destination_rank,
            )
            for route in self.graph.cross_routes
        )
        endpoint_pairs = sorted(
            {
                (flow.source_rank, flow.destination_rank)
                for handoff in self.pd_plan.handoffs
                for flow in handoff.flows
            }
        )
        expected_pairs = tuple(
            (
                self.pd_plan.prefill_instance_ref,
                source_rank,
                self.pd_plan.decode_instance_ref,
                destination_rank,
            )
            for source_rank, destination_rank in endpoint_pairs
        )
        if actual_pairs != expected_pairs:
            raise SchemaError("cross routes must exactly cover PD flow endpoints", path=f"{path}.graph.cross_routes")
        expected_id = stable_artifact_id(
            "stage4_placed_ir1",
            self._semantic_key(),
            schema_version=STAGE4_PLACED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: IR0,
        context: PlacementContext,
        path: str = "stage4_placed_ir1",
    ) -> None:
        if type(source) is not IR0:
            raise SchemaError("must be an IR0", path="source")
        if type(context) is not PlacementContext:
            raise SchemaError("must be a PlacementContext", path="placement_context")
        source.validate("source")
        context.validate("placement_context")
        self.validate(path)
        if self.source_ir0_id != source.id:
            raise SchemaError("must match source IR0", path=f"{path}.source_ir0_id")
        if self.placement_context_id != context.id:
            raise SchemaError("must match placement context", path=f"{path}.placement_context_id")
        if source.pd_plan_id != self.pd_plan.id:
            raise SchemaError("source must reference embedded PD plan", path=f"{path}.pd_plan")
        if self.graph.fabric != context.fabric:
            raise SchemaError("must use placement-context fabric", path=f"{path}.graph.fabric")
        for field_name in (
            "profile", "values", "edges", "fusion_candidates", "state_accesses",
            "instance_profiles", "node_profiles", "pd_plan_id",
        ):
            if getattr(self.graph, field_name) != getattr(source, field_name):
                raise SchemaError("must exactly preserve source IR0 field", path=f"{path}.graph.{field_name}")
        if self.graph.persistent_state_manifest is None:
            if source.persistent_states:
                raise SchemaError("persistent state requires HBM backing", path=f"{path}.graph.persistent_state_manifest")
        else:
            manifest = self.graph.persistent_state_manifest
            if manifest.declarations != source.persistent_states:
                raise SchemaError("must preserve state declarations", path=f"{path}.graph.persistent_state_manifest.declarations")
            if manifest.address_spaces != context.hbm_address_spaces:
                raise SchemaError("must preserve context HBM spaces", path=f"{path}.graph.persistent_state_manifest.address_spaces")
        expected_group_sources = _expected_groups(source.instances)
        if len(self.graph.groups) != len(expected_group_sources):
            raise SchemaError("must contain one group per logical mesh", path=f"{path}.graph.groups")
        for index, (group, (instance, mesh)) in enumerate(zip(self.graph.groups, expected_group_sources)):
            _validate_group_against_mesh(group, instance, mesh, path=f"{path}.graph.groups[{index}]")
        group_by_owner = {(group.instance_id, group.mesh_ref): group for group in self.graph.groups}
        if len(self.graph.instances) != len(source.instances):
            raise SchemaError("must preserve every logical instance", path=f"{path}.graph.instances")
        for index, (physical, logical) in enumerate(zip(self.graph.instances, source.instances)):
            owned = tuple(group for group in self.graph.groups if group.instance_id == logical.id)
            _validate_instance_against_logical(physical, logical, source, owned, path=f"{path}.graph.instances[{index}]")
        if len(self.graph.nodes) != len(source.nodes):
            raise SchemaError("must preserve every logical node", path=f"{path}.graph.nodes")
        for index, (physical, logical) in enumerate(zip(self.graph.nodes, source.nodes)):
            _validate_node_against_logical(physical, logical, group_by_owner, path=f"{path}.graph.nodes[{index}]")
        if context.placement.strategy is PlacementStrategy.EXPLICIT:
            expected = {(item.instance_id, item.mesh_ref): item.die_ids for item in context.placement.groups}
            actual = {
                (group.instance_id, group.mesh_ref): tuple(item.die_id for item in group.placements)
                for group in self.graph.groups
            }
            if actual != expected:
                raise SchemaError("must exactly match explicit placement", path=f"{path}.graph.groups")


__all__ = [
    "PLACED_IR1_BUNDLE_SCHEMA_VERSION",
    "STAGE4_PLACED_IR1_SCHEMA_VERSION",
    "TRAIN_PLACED_IR1_SCHEMA_VERSION",
    "PlacedIR1Bundle",
    "PlacedProfileIR1",
    "Stage4PlacedIR1",
    "TrainPlacedIR1",
    "TrainPlacedReplica",
]
