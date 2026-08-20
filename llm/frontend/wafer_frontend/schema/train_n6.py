"""Dedicated DP-replica N6 lowering carrier for forward-only training."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from .artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    RegionManifest,
)
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import TrainStructure
from .n6 import _leaf_fragment, _validate_lowered_fragments
from .train_global_action import (
    TrainGlobalAction,
    TrainGlobalActionReplica,
)


TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.train_lowered_program/v1alpha3"
)
TRAIN_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.train_linked_program/v1alpha3"
)


def train_replica_lowering_context(
    source: TrainGlobalActionReplica,
) -> LoweringContext:
    """Recover the one lossless lowering context owned by a DP replica."""

    if type(source) is not TrainGlobalActionReplica:
        raise SchemaError(
            "must be a TrainGlobalActionReplica",
            path="source",
        )
    source.validate("source")
    projected = source.scheduled.projected
    context = LoweringContext(
        ir1=projected.graph,
        fusion_plans=projected.fusion_plans,
        standalone_plans=projected.standalone_plans,
        projection=projected.projection,
        schedule_set=source.scheduled.schedule_set,
        global_dag=source.global_dag,
    )
    context.validate("lowering_context")
    return context


@dataclass(frozen=True, slots=True)
class TrainLoweredReplica:
    """Canonical lowering leaves for exactly one DP replica."""

    id: str
    replica_index: int
    source_global_action_replica_id: str
    source_scheduled_replica_id: str
    source_projected_replica_id: str
    source_replica_plan_id: str
    source_ir1_id: str
    lowering_context: LoweringContext
    fragments: tuple[CommandFragment | RegionManifest, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainGlobalActionReplica,
        lowering_context: LoweringContext,
        fragments: tuple[CommandFragment | RegionManifest, ...],
    ) -> "TrainLoweredReplica":
        canonical_fragments = tuple(
            sorted(fragments, key=lambda item: _leaf_fragment(item).id)
        )
        projected = source.scheduled.projected
        semantic_key = {
            "replica_index": source.replica_index,
            "source_global_action_replica_id": source.id,
            "source_scheduled_replica_id": source.scheduled.id,
            "source_projected_replica_id": projected.id,
            "source_replica_plan_id": projected.source_replica_plan_id,
            "source_ir1_id": projected.graph.id,
            "lowering_context": lowering_context,
            "fragments": canonical_fragments,
        }
        result = cls(
            id=stable_artifact_id(
                "train_lowered_replica",
                semantic_key,
                schema_version=TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "replica_index": self.replica_index,
            "source_global_action_replica_id": (
                self.source_global_action_replica_id
            ),
            "source_scheduled_replica_id": self.source_scheduled_replica_id,
            "source_projected_replica_id": self.source_projected_replica_id,
            "source_replica_plan_id": self.source_replica_plan_id,
            "source_ir1_id": self.source_ir1_id,
            "lowering_context": self.lowering_context,
            "fragments": self.fragments,
        }

    def validate(self, path: str = "train_lowered_replica") -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        for field_name in (
            "source_global_action_replica_id",
            "source_scheduled_replica_id",
            "source_projected_replica_id",
            "source_replica_plan_id",
            "source_ir1_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.lowering_context) is not LoweringContext:
            raise SchemaError(
                "must be a LoweringContext",
                path=f"{path}.lowering_context",
            )
        self.lowering_context.validate(f"{path}.lowering_context")
        if (
            self.source_ir1_id != self.lowering_context.ir1.id
            or self.lowering_context.ir1.producer_pass != "fusion_partition"
            or len(self.lowering_context.ir1.groups) != 1
            or not self.lowering_context.ir1.groups[0].id.endswith(
                f"__dp{self.replica_index}"
            )
            or self.lowering_context.projection.state_transfers
            or self.lowering_context.projection.producer_pass
            != "project_to_ir2"
            or self.lowering_context.schedule_set.producer_pass
            != "intra_die_schedule"
            or self.lowering_context.global_dag.producer_pass
            != "global_action_dag"
        ):
            raise SchemaError(
                "must preserve one exact forward-train replica context",
                path=f"{path}.lowering_context",
            )
        _validate_lowered_fragments(
            self.fragments,
            self.lowering_context,
            f"{path}.fragments",
        )
        expected_id = stable_artifact_id(
            "train_lowered_replica",
            self._semantic_key(),
            schema_version=TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable replica id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainGlobalActionReplica,
        path: str = "train_lowered_replica",
    ) -> None:
        if type(source) is not TrainGlobalActionReplica:
            raise SchemaError(
                "must be a TrainGlobalActionReplica",
                path="source",
            )
        source.validate("source")
        self.validate(path)
        projected = source.scheduled.projected
        if (
            self.replica_index != source.replica_index
            or self.source_global_action_replica_id != source.id
            or self.source_scheduled_replica_id != source.scheduled.id
            or self.source_projected_replica_id != projected.id
            or self.source_replica_plan_id != projected.source_replica_plan_id
            or self.source_ir1_id != projected.graph.id
            or self.lowering_context != train_replica_lowering_context(source)
        ):
            raise SchemaError(
                "must preserve complete replica lowering provenance",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TrainLoweredProgram:
    """Canonical pre-link lowering product for all DP replicas."""

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
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainLoweredReplica, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainGlobalAction,
        replicas: tuple[TrainLoweredReplica, ...],
    ) -> "TrainLoweredProgram":
        semantic_key = {
            "source_global_action_carrier_id": source.id,
            "source_scheduled_carrier_id": source.source_scheduled_carrier_id,
            "source_projected_carrier_id": source.source_projected_carrier_id,
            "source_planned_carrier_id": source.source_planned_carrier_id,
            "source_partitioned_carrier_id": (
                source.source_partitioned_carrier_id
            ),
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": source.planning_context_id,
            "projection_context_id": source.projection_context_id,
            "scheduling_context_id": source.scheduling_context_id,
            "train_structure": source.train_structure,
            "dp_degree": source.dp_degree,
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            producer_pass="train_lowering",
            id=stable_artifact_id(
                "train_lowered_program",
                semantic_key,
                schema_version=TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
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
                "train_structure",
                "dp_degree",
                "replicas",
            )
        }

    def validate(self, path: str = "train_lowered_program") -> None:
        if self.schema_version != TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "train_lowering":
            raise SchemaError(
                "must be 'train_lowering'",
                path=f"{path}.producer_pass",
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
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError(
                "must be a TrainStructure",
                path=f"{path}.train_structure",
            )
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError(
                "must contain exact DP replica lowering coverage",
                path=f"{path}.replicas",
            )
        if tuple(item.replica_index for item in self.replicas) != tuple(
            range(self.dp_degree)
        ):
            raise SchemaError(
                "replicas must use canonical DP order",
                path=f"{path}.replicas",
            )

        namespaces = tuple(set() for _ in range(10))
        reference_fabric = None
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainLoweredReplica:
                raise SchemaError(
                    "must be a TrainLoweredReplica",
                    path=replica_path,
                )
            replica.validate(replica_path)
            context = replica.lowering_context
            leaves = tuple(_leaf_fragment(item) for item in replica.fragments)
            local_namespaces = (
                {replica.source_global_action_replica_id},
                {context.ir1.id},
                {context.projection.id},
                {context.schedule_set.id},
                {context.global_dag.id},
                {leaf.id for leaf in leaves},
                {
                    action_id
                    for leaf in leaves
                    for action_id in leaf.claimed_action_ids
                },
                {
                    (action.logical_core.die_id, action.logical_core.local_core_id)
                    for action in context.global_dag.actions
                    if action.logical_core is not None
                },
                {abi.id for leaf in leaves for abi in leaf.buffer_abi},
                {abi.id for leaf in leaves for abi in leaf.state_abi},
            )
            if any(
                namespace.intersection(local)
                for namespace, local in zip(namespaces, local_namespaces)
            ):
                raise SchemaError(
                    "replica contexts, actions, cores, leaves, and ABIs must be disjoint",
                    path=replica_path,
                )
            for namespace, local in zip(namespaces, local_namespaces):
                namespace.update(local)
            if reference_fabric is None:
                reference_fabric = context.ir1.fabric
            elif context.ir1.fabric != reference_fabric:
                raise SchemaError(
                    "all replicas must share one physical fabric",
                    path=f"{replica_path}.lowering_context.ir1.fabric",
                )

        expected_id = stable_artifact_id(
            "train_lowered_program",
            self._semantic_key(),
            schema_version=TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainGlobalAction,
        path: str = "train_lowered_program",
    ) -> None:
        if type(source) is not TrainGlobalAction:
            raise SchemaError("must be a TrainGlobalAction", path="source")
        source.validate("source")
        self.validate(path)
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
            "train_structure",
            "dp_degree",
        ):
            if getattr(self, field_name) != getattr(source, field_name):
                raise SchemaError(
                    "must preserve complete train GlobalAction provenance",
                    path=f"{path}.{field_name}",
                )
        if self.source_global_action_carrier_id != source.id:
            raise SchemaError(
                "must match source GlobalAction carrier",
                path=f"{path}.source_global_action_carrier_id",
            )
        if len(self.replicas) != len(source.replicas):
            raise SchemaError(
                "must preserve exact source replica coverage",
                path=f"{path}.replicas",
            )
        for index, (replica, source_replica) in enumerate(
            zip(self.replicas, source.replicas)
        ):
            replica.validate_against(
                source_replica,
                f"{path}.replicas[{index}]",
            )


@dataclass(frozen=True, slots=True)
class TrainLinkedProgram:
    """One executable manifest for the exact multi-replica Train lowering."""

    schema_version: str
    producer_pass: str
    id: str
    source: TrainLoweredProgram
    manifest: LinkedProgramManifest

    @classmethod
    def create(
        cls,
        *,
        source: TrainLoweredProgram,
        manifest: LinkedProgramManifest,
    ) -> "TrainLinkedProgram":
        semantic_key = {
            "source": source,
            "manifest": manifest,
        }
        result = cls(
            schema_version=TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            producer_pass="train_manifest_linker",
            id=stable_artifact_id(
                "train_linked_program",
                semantic_key,
                schema_version=TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"source": self.source, "manifest": self.manifest}

    def validate(self, path: str = "train_linked_program") -> None:
        if self.schema_version != TRAIN_LINKED_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "train_manifest_linker":
            raise SchemaError(
                "must be 'train_manifest_linker'",
                path=f"{path}.producer_pass",
            )
        if type(self.source) is not TrainLoweredProgram:
            raise SchemaError(
                "must embed one exact TrainLoweredProgram",
                path=f"{path}.source",
            )
        self.source.validate(f"{path}.source")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError(
                "must embed one LinkedProgramManifest",
                path=f"{path}.manifest",
            )
        self.manifest.validate(f"{path}.manifest")
        if (
            self.manifest.source_ir1_id
            != self.source.source_planned_carrier_id
            or self.manifest.source_projection_id
            != self.source.source_projected_carrier_id
            or self.manifest.source_schedule_set_id
            != self.source.source_scheduled_carrier_id
            or self.manifest.source_global_dag_id
            != self.source.source_global_action_carrier_id
        ):
            raise SchemaError(
                "manifest top ids must equal the Train quotient carrier ids",
                path=f"{path}.manifest",
            )
        expected_fragments = tuple(
            sorted(
                (
                    fragment
                    for replica in self.source.replicas
                    for fragment in replica.fragments
                ),
                key=lambda item: item.id,
            )
        )
        if self.manifest.fragments != expected_fragments:
            raise SchemaError(
                "manifest must consume every replica fragment exactly once",
                path=f"{path}.manifest.fragments",
            )
        from ..lowering.linker import NaiveManifestLinker

        expected_manifest = NaiveManifestLinker().link_train(self.source)
        if self.manifest != expected_manifest:
            raise SchemaError(
                "manifest must equal the strict unified Train quotient",
                path=f"{path}.manifest",
            )
        expected_id = stable_artifact_id(
            "train_linked_program",
            self._semantic_key(),
            schema_version=TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable linked Train id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainLoweredProgram,
        path: str = "train_linked_program",
    ) -> None:
        if type(source) is not TrainLoweredProgram:
            raise SchemaError(
                "must be a TrainLoweredProgram",
                path="source",
            )
        self.validate(path)
        if self.source != source:
            raise SchemaError(
                "must embed the exact lowering source",
                path=f"{path}.source",
            )


__all__ = [
    "TRAIN_LINKED_PROGRAM_SCHEMA_VERSION",
    "TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION",
    "TrainLinkedProgram",
    "TrainLoweredProgram",
    "TrainLoweredReplica",
    "train_replica_lowering_context",
]
