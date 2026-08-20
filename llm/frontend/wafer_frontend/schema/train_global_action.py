"""Exact DP-replica carrier for forward-train GlobalAction quotients."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import GlobalActionDAG
from .ir0 import (
    CrossEntropyBackwardWorkload,
    CrossEntropyForwardWorkload,
    OpKind,
    OpPhase,
    SgdUpdateWorkload,
    TrainStructure,
)
from .ir2 import (
    BufferAccess,
    BufferUseRole,
    SemanticTaskKind,
    StateUseAccess,
)
from .n5 import TrainScheduledIR2, TrainScheduledReplica
from .persistent_state import (
    PersistentStateAccess,
    StateKind,
)


TRAIN_GLOBAL_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.train_global_action/v1alpha1"
)
S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_train_global_action/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class TrainGlobalActionReplica:
    """One DP replica and its exact one-action-per-task quotient."""

    id: str
    replica_index: int
    source_scheduled_replica_id: str
    scheduled: TrainScheduledReplica
    global_dag: GlobalActionDAG

    @classmethod
    def create(
        cls,
        *,
        source: TrainScheduledReplica,
        global_dag: GlobalActionDAG,
    ) -> "TrainGlobalActionReplica":
        semantic_key = {
            "replica_index": source.replica_index,
            "source_scheduled_replica_id": source.id,
            "scheduled_id": source.id,
            "global_dag_id": global_dag.id,
        }
        result = cls(
            id=stable_artifact_id(
                "train_global_action_replica",
                semantic_key,
                schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            replica_index=source.replica_index,
            source_scheduled_replica_id=source.id,
            scheduled=source,
            global_dag=global_dag,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "replica_index": self.replica_index,
            "source_scheduled_replica_id": self.source_scheduled_replica_id,
            "scheduled_id": self.scheduled.id,
            "global_dag_id": self.global_dag.id,
        }

    def validate(self, path: str = "train_global_action_replica") -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        validate_nonempty(
            self.source_scheduled_replica_id,
            f"{path}.source_scheduled_replica_id",
        )
        if type(self.scheduled) is not TrainScheduledReplica:
            raise SchemaError(
                "must be a TrainScheduledReplica",
                path=f"{path}.scheduled",
            )
        self.scheduled.validate(f"{path}.scheduled")
        if (
            self.replica_index != self.scheduled.replica_index
            or self.source_scheduled_replica_id != self.scheduled.id
        ):
            raise SchemaError(
                "must preserve scheduled replica identity",
                path=path,
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
        projected = self.scheduled.projected
        self.global_dag.validate_against(
            projected.graph,
            projected.projection,
            self.scheduled.schedule_set,
            f"{path}.global_dag",
        )

        graph = projected.graph
        manifest = graph.persistent_state_manifest
        if manifest is None or any(
            declaration.identity.kind is not StateKind.PARAMETER
            for declaration in manifest.declarations
        ):
            raise SchemaError(
                "forward-train GlobalAction requires parameter-only persistent state",
                path=f"{path}.scheduled.projected.graph.persistent_state_manifest",
            )
        state_actions = tuple(
            action for action in self.global_dag.actions if action.state_uses
        )
        if any(
            action.task_kind is not SemanticTaskKind.DMA_IN
            or len(action.state_uses) != 1
            or action.state_uses[0].access is not StateUseAccess.READ
            for action in state_actions
        ):
            raise SchemaError(
                "forward-train state actions must be READ-only DMA_IN",
                path=f"{path}.global_dag.actions",
            )
        expected_hbm_bindings = {binding.id for binding in manifest.bindings}
        actual_hbm_bindings = tuple(
            action.state_uses[0].hbm_binding_ref for action in state_actions
        )
        if (
            len(actual_hbm_bindings) != len(set(actual_hbm_bindings))
            or set(actual_hbm_bindings) != expected_hbm_bindings
            or sum(action.bytes for action in state_actions)
            != sum(binding.size_bytes for binding in manifest.bindings)
        ):
            raise SchemaError(
                "HBM READ actions must exactly cover parameter bindings and bytes",
                path=f"{path}.global_dag.actions",
            )

        group = graph.groups[0]
        local_route_ids = {route.id for route in group.embedding.routes}
        if any(
            action.flow_route is not None
            and action.flow_route.pair_route_ref not in local_route_ids
            for action in self.global_dag.actions
        ):
            raise SchemaError(
                "collective action route cannot escape its DP replica group",
                path=f"{path}.global_dag.actions",
            )

        ce_node = next(
            (node for node in graph.nodes if node.kind is OpKind.CE_FORWARD),
            None,
        )
        if ce_node is None or type(ce_node.workload) is not CrossEntropyForwardWorkload:
            raise SchemaError(
                "requires one typed CE_FORWARD node",
                path=f"{path}.scheduled.projected.graph.nodes",
            )
        incoming = tuple(
            edge for edge in graph.edges if edge.destination_node == ce_node.id
        )
        if len(incoming) != 1:
            raise SchemaError(
                "CE_FORWARD requires one LM-head predecessor",
                path=f"{path}.scheduled.projected.graph.edges",
            )
        predecessor_id = incoming[0].source_node
        actions_by_op_rank: dict[tuple[str, int], object] = {}
        for action in self.global_dag.actions:
            origin = action.origin_ref
            op_id = getattr(origin, "op_id", None)
            rank = getattr(origin, "rank", None)
            if op_id is not None and rank is not None:
                actions_by_op_rank[(op_id, rank)] = action
        ce_actions = tuple(
            action
            for action in self.global_dag.actions
            if action.op_kind is OpKind.CE_FORWARD
        )
        if len(ce_actions) != len(group.placements):
            raise SchemaError(
                "CE_FORWARD must have one GlobalAction per TP rank",
                path=f"{path}.global_dag.actions",
            )
        rank_by_die = {
            placement.die_id: placement.rank for placement in group.placements
        }
        for action in ce_actions:
            if (
                action.task_kind is not SemanticTaskKind.COMP
                or action.compute is None
                or type(action.compute.workload)
                is not CrossEntropyForwardWorkload
                or action.logical_core is None
                or action.logical_core.die_id not in rank_by_die
            ):
                raise SchemaError(
                    "CE_FORWARD action semantics are not exact",
                    path=f"{path}.global_dag.actions",
                )
            rank = rank_by_die[action.logical_core.die_id]
            predecessor = actions_by_op_rank.get((predecessor_id, rank))
            if predecessor is None or action.deps != (predecessor.id,):
                raise SchemaError(
                    "CE_FORWARD must directly and exclusively depend on its rank-local LM head",
                    path=f"{path}.global_dag.actions",
                )
            if (
                predecessor.logical_core != action.logical_core
                or predecessor.core_order_index is None
                or action.core_order_index != predecessor.core_order_index + 1
            ):
                raise SchemaError(
                    "LM head must immediately precede CE_FORWARD on one logical core",
                    path=f"{path}.global_dag.actions",
                )
            if tuple(
                (use.role, use.access, use.operand_index)
                for use in action.buffer_uses
            ) != (
                (BufferUseRole.COMP_INPUT, BufferAccess.READ, 0),
                (BufferUseRole.COMP_INPUT, BufferAccess.READ, 1),
                (BufferUseRole.COMP_OUTPUT, BufferAccess.WRITE, 0),
            ):
                raise SchemaError(
                    "CE_FORWARD GlobalAction buffer roles/arity are not exact",
                    path=f"{path}.global_dag.actions",
                )

        expected_id = stable_artifact_id(
            "train_global_action_replica",
            self._semantic_key(),
            schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable replica id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainScheduledReplica,
        path: str = "train_global_action_replica",
    ) -> None:
        source.validate("source")
        self.validate(path)
        if self.scheduled != source or self.source_scheduled_replica_id != source.id:
            raise SchemaError(
                "must preserve the exact scheduled replica",
                path=path,
            )


@dataclass(frozen=True, slots=True)
class TrainGlobalAction:
    """Canonical DP quotient over exact per-replica GlobalAction DAGs."""

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
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainGlobalActionReplica, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainScheduledIR2,
        replicas: tuple[TrainGlobalActionReplica, ...],
    ) -> "TrainGlobalAction":
        semantic_key = {
            "source_scheduled_carrier_id": source.id,
            "source_projected_carrier_id": source.source_projected_carrier_id,
            "source_planned_carrier_id": source.source_planned_carrier_id,
            "source_partitioned_carrier_id": source.source_partitioned_carrier_id,
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
            schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            producer_pass="train_global_action",
            id=stable_artifact_id(
                "train_global_action",
                semantic_key,
                schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_scheduled_carrier_id": self.source_scheduled_carrier_id,
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

    def validate(self, path: str = "train_global_action") -> None:
        if self.schema_version != TRAIN_GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "train_global_action":
            raise SchemaError(
                "must be 'train_global_action'",
                path=f"{path}.producer_pass",
            )
        for name in (
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
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError(
                "must be a TrainStructure",
                path=f"{path}.train_structure",
            )
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError(
                "must contain exact DP replica GlobalAction coverage",
                path=f"{path}.replicas",
            )
        if tuple(replica.replica_index for replica in self.replicas) != tuple(
            range(self.dp_degree)
        ):
            raise SchemaError(
                "replica GlobalAction quotients must use canonical DP order",
                path=f"{path}.replicas",
            )

        action_ids: set[str] = set()
        dag_ids: set[str] = set()
        schedule_ids: set[str] = set()
        logical_cores: set[tuple[int, int]] = set()
        hbm_binding_ids: set[str] = set()
        flow_ids: set[str] = set()
        route_ids: set[str] = set()
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainGlobalActionReplica:
                raise SchemaError(
                    "must be a TrainGlobalActionReplica",
                    path=replica_path,
                )
            replica.validate(replica_path)
            local_actions = {action.id for action in replica.global_dag.actions}
            local_dags = {ref.dag_id for ref in replica.global_dag.scheduled_dags}
            local_schedules = {
                ref.schedule_id for ref in replica.global_dag.scheduled_dags
            }
            local_cores = {
                (action.logical_core.die_id, action.logical_core.local_core_id)
                for action in replica.global_dag.actions
                if action.logical_core is not None
            }
            local_hbm = {
                use.hbm_binding_ref
                for action in replica.global_dag.actions
                for use in action.state_uses
            }
            local_flows = {
                action.flow_id
                for action in replica.global_dag.actions
                if action.flow_id is not None
            }
            local_routes = {
                action.flow_route.pair_route_ref
                for action in replica.global_dag.actions
                if action.flow_route is not None
            }
            intersections = (
                action_ids.intersection(local_actions),
                dag_ids.intersection(local_dags),
                schedule_ids.intersection(local_schedules),
                logical_cores.intersection(local_cores),
                hbm_binding_ids.intersection(local_hbm),
                flow_ids.intersection(local_flows),
                route_ids.intersection(local_routes),
            )
            if any(intersections):
                raise SchemaError(
                    "action, schedule, core, HBM, flow, and route identities must be DP-replica-disjoint",
                    path=replica_path,
                )
            action_ids.update(local_actions)
            dag_ids.update(local_dags)
            schedule_ids.update(local_schedules)
            logical_cores.update(local_cores)
            hbm_binding_ids.update(local_hbm)
            flow_ids.update(local_flows)
            route_ids.update(local_routes)

        expected_id = stable_artifact_id(
            "train_global_action",
            self._semantic_key(),
            schema_version=TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainScheduledIR2,
        path: str = "train_global_action",
    ) -> None:
        if type(source) is not TrainScheduledIR2:
            raise SchemaError("must be a TrainScheduledIR2", path="source")
        source.validate("source")
        self.validate(path)
        if (
            self.source_scheduled_carrier_id != source.id
            or self.source_projected_carrier_id
            != source.source_projected_carrier_id
            or self.source_planned_carrier_id != source.source_planned_carrier_id
            or self.source_partitioned_carrier_id
            != source.source_partitioned_carrier_id
            or self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
            or self.planning_context_id != source.planning_context_id
            or self.projection_context_id != source.projection_context_id
            or self.scheduling_context_id != source.scheduling_context_id
            or self.train_structure != source.train_structure
            or self.dp_degree != source.dp_degree
            or len(self.replicas) != len(source.replicas)
        ):
            raise SchemaError(
                "must preserve complete train scheduling provenance",
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
class S2LiteTrainGlobalAction:
    """One exact TP1/DP1 LM-head-only backward/update GlobalAction artifact."""

    schema_version: str
    producer_pass: str
    id: str
    source: TrainScheduledIR2
    global_dags: tuple[GlobalActionDAG, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainScheduledIR2,
        global_dags: tuple[GlobalActionDAG, ...],
    ) -> "S2LiteTrainGlobalAction":
        semantic_key = {
            "source": source,
            "global_dags": global_dags,
        }
        result = cls(
            schema_version=S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            producer_pass="s2_lite_train_global_action",
            id=stable_artifact_id(
                "s2_lite_train_global_action",
                semantic_key,
                schema_version=S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source": self.source,
            "global_dags": self.global_dags,
        }

    def validate(self, path: str = "s2_lite_train_global_action") -> None:
        if self.schema_version != S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "s2_lite_train_global_action":
            raise SchemaError(
                "must be 's2_lite_train_global_action'",
                path=f"{path}.producer_pass",
            )
        if type(self.source) is not TrainScheduledIR2:
            raise SchemaError("must be a TrainScheduledIR2", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if self.source.dp_degree != 1 or len(self.source.replicas) != 1:
            raise SchemaError(
                "S2-Lite GlobalAction requires DP1",
                path=f"{path}.source.dp_degree",
            )
        if len(self.global_dags) != 1:
            raise SchemaError(
                "must contain one DP1 GlobalAction DAG",
                path=f"{path}.global_dags",
            )
        replica = self.source.replicas[0]
        dag = self.global_dags[0]
        if type(dag) is not GlobalActionDAG:
            raise SchemaError(
                "must be a GlobalActionDAG", path=f"{path}.global_dags[0]"
            )
        dag.validate_against(
            replica.projected.graph,
            replica.projected.projection,
            replica.schedule_set,
            f"{path}.global_dags[0]",
        )
        graph = replica.projected.graph
        if len(dag.actions) != 46:
            raise SchemaError(
                "S2-Lite tiny case requires exactly 46 actions",
                path=f"{path}.global_dags[0].actions",
            )
        manifest = graph.persistent_state_manifest
        if manifest is None:
            raise SchemaError(
                "requires persistent state",
                path=f"{path}.source.replicas[0].projected.graph.persistent_state_manifest",
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
            or len(frozen) != 14
            or trainable[0].access is not PersistentStateAccess.READ_WRITE
            or any(
                declaration.access is not PersistentStateAccess.READ_ONLY
                for declaration in frozen
            )
        ):
            raise SchemaError(
                "requires one READ_WRITE trainable and fourteen frozen parameters",
                path=f"{path}.source.replicas[0].projected.graph.persistent_state_manifest.declarations",
            )
        binding_by_state = {
            binding.state_ref: binding.id for binding in manifest.bindings
        }
        trainable_binding = binding_by_state[trainable[0].id]
        state_actions = tuple(action for action in dag.actions if action.state_uses)
        dma_in = tuple(
            action
            for action in state_actions
            if action.task_kind is SemanticTaskKind.DMA_IN
        )
        dma_out = tuple(
            action
            for action in state_actions
            if action.task_kind is SemanticTaskKind.DMA_OUT
        )
        if (
            len(state_actions) != 17
            or len(dma_in) != 16
            or len(dma_out) != 1
            or dma_out[0].state_uses[0].access is not StateUseAccess.WRITE
            or dma_out[0].state_uses[0].hbm_binding_ref != trainable_binding
            or sum(
                1
                for action in dma_in
                if action.state_uses[0].hbm_binding_ref == trainable_binding
                and action.state_uses[0].access is StateUseAccess.READ
            )
            != 2
            or any(
                sum(
                    1
                    for action in dma_in
                    if action.state_uses[0].hbm_binding_ref
                    == binding_by_state[declaration.id]
                    and action.state_uses[0].access is StateUseAccess.READ
                )
                != 1
                for declaration in frozen
            )
        ):
            raise SchemaError(
                "state actions must exactly cover frozen reads and one trainable read/write cycle",
                path=f"{path}.global_dags[0].actions",
            )

        node_by_id = {node.id: node for node in graph.nodes}
        compute_actions = tuple(
            action for action in dag.actions if action.task_kind is SemanticTaskKind.COMP
        )
        ce_backward = tuple(
            action for action in compute_actions if action.op_kind is OpKind.CE_BACKWARD
        )
        optimizer = tuple(
            action
            for action in compute_actions
            if action.op_kind is OpKind.OPTIMIZER_UPDATE
        )
        wgrad = tuple(
            action
            for action in compute_actions
            if action.op_kind is OpKind.GEMM
            and action.member_id is not None
            and node_by_id[action.member_id].phase is OpPhase.WGRAD
        )
        if (
            len(ce_backward) != 1
            or ce_backward[0].compute is None
            or type(ce_backward[0].compute.workload)
            is not CrossEntropyBackwardWorkload
            or len(wgrad) != 1
            or len(optimizer) != 1
            or optimizer[0].compute is None
            or type(optimizer[0].compute.workload) is not SgdUpdateWorkload
        ):
            raise SchemaError(
                "must contain one exact CE_BACKWARD, WGRAD, and SGD action",
                path=f"{path}.global_dags[0].actions",
            )
        expected_id = stable_artifact_id(
            "s2_lite_train_global_action",
            self._semantic_key(),
            schema_version=S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: TrainScheduledIR2,
        path: str = "s2_lite_train_global_action",
    ) -> None:
        self.validate(path)
        if self.source != source:
            raise SchemaError(
                "must preserve the exact scheduled source", path=f"{path}.source"
            )


__all__ = [
    "S2_LITE_TRAIN_GLOBAL_ACTION_SCHEMA_VERSION",
    "S2LiteTrainGlobalAction",
    "TRAIN_GLOBAL_ACTION_SCHEMA_VERSION",
    "TrainGlobalAction",
    "TrainGlobalActionReplica",
]
