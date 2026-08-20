"""Typed DP4xTP1 2x2-tree AllReduce carriers for S2-Lite."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_uint64
from .global_action import GlobalAction, GlobalActionDAG
from .ir0 import IR0, OpKind
from .lite_train_graph import S2LiteLmHeadTrainIR0
from .n5 import TrainScheduledIR2
from .persistent_state import PersistentStateAccess, StateKind


S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp4_tree_ar_source/v1alpha1"
)
S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp4_tree_ar_global_action/v1alpha1"
)
S2_LITE_DP4_TREE_AR_CASE_ID = "case.s2_lite.dp4_tp1.tree_allreduce"


class TreeArFlowKind(str, Enum):
    UPLOAD_1_TO_0 = "upload_1_to_0"
    UPLOAD_3_TO_2 = "upload_3_to_2"
    PARTIAL_2_TO_0 = "partial_2_to_0"
    BROADCAST_0_TO_1 = "broadcast_0_to_1"
    BROADCAST_0_TO_2 = "broadcast_0_to_2"
    BROADCAST_2_TO_3 = "broadcast_2_to_3"


class TreeArReduceKind(str, Enum):
    PAIR_01 = "pair_01"
    PAIR_23 = "pair_23"
    GLOBAL_AT_0 = "global_at_0"


def _dp4_graph(base: S2LiteLmHeadTrainIR0) -> IR0:
    instance = base.graph.instances[0]
    replicated = replace(instance, parallel=replace(instance.parallel, dp=4))
    return IR0.create(
        producer_pass="s2_lite_dp4_tree_ar_source",
        job=base.graph.job,
        instances=(replicated,),
        nodes=base.graph.nodes,
        values=base.graph.values,
        edges=base.graph.edges,
        fusion_candidates=base.graph.fusion_candidates,
        profile=base.graph.profile,
        train=base.graph.train,
        instance_profiles=base.graph.instance_profiles,
        node_profiles=base.graph.node_profiles,
        pd_plan_id=base.graph.pd_plan_id,
        persistent_states=base.graph.persistent_states,
        state_accesses=base.graph.state_accesses,
    )


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArSource:
    schema_version: str
    producer_pass: str
    id: str
    base: S2LiteLmHeadTrainIR0
    graph: IR0

    @classmethod
    def create(cls, *, base: S2LiteLmHeadTrainIR0) -> "S2LiteDp4TreeArSource":
        semantic_key = {"base": base, "graph": _dp4_graph(base)}
        result = cls(
            schema_version=S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION,
            producer_pass="s2_lite_dp4_tree_ar_source",
            id=stable_artifact_id(
                "s2_lite_dp4_tree_ar_source",
                semantic_key,
                schema_version=S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"base": self.base, "graph": self.graph}

    def validate(self, path: str = "s2_lite_dp4_tree_ar_source") -> None:
        if self.schema_version != S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp4_tree_ar_source":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        if type(self.base) is not S2LiteLmHeadTrainIR0:
            raise SchemaError("must embed the exact Lite template", path=f"{path}.base")
        self.base.validate(f"{path}.base")
        self.graph.validate(f"{path}.graph")
        if self.graph != _dp4_graph(self.base):
            raise SchemaError("graph must be the exact DP4 replication", path=f"{path}.graph")
        instance = self.graph.instances[0]
        if (
            instance.parallel.dp,
            instance.parallel.tp,
            instance.parallel.pp,
            instance.parallel.ep,
            len(self.graph.nodes),
            len(self.graph.persistent_states),
            len(self.graph.state_accesses),
        ) != (4, 1, 1, 1, 29, 15, 16):
            raise SchemaError("requires exact DP4xTP1 Lite graph", path=f"{path}.graph")
        expected = stable_artifact_id(
            "s2_lite_dp4_tree_ar_source",
            self._semantic_key(),
            schema_version=S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteDp4ScratchRoot:
    die_id: int
    span_bytes: int
    slice_offsets_bytes: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.die_id, f"{path}.die_id")
        validate_uint64(self.span_bytes, f"{path}.span_bytes")
        if self.die_id not in (0, 2) or self.span_bytes != 4096 or self.slice_offsets_bytes != (0, 2048):
            raise SchemaError("requires exact die0/die2 packed FP32 scratch", path=path)


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArContract:
    case_id: str
    replica_die_ids: tuple[int, ...]
    gradient_dtype: DType
    gradient_bytes: int
    logical_flow_bytes: int
    reduce_input_count: int
    reduce_element_count: int
    input_stride_bytes: int
    reduce_source_span_bytes: int
    reduce_destination_span_bytes: int
    alignment_bytes: int
    scratch_roots: tuple[S2LiteDp4ScratchRoot, ...]

    def validate(self, path: str) -> None:
        if (
            self.case_id != S2_LITE_DP4_TREE_AR_CASE_ID
            or self.replica_die_ids != (0, 1, 2, 3)
            or self.gradient_dtype is not DType.FP32
            or self.gradient_bytes != 2048
            or self.logical_flow_bytes != 12288
            or self.reduce_input_count != 2
            or self.reduce_element_count != 512
            or self.input_stride_bytes != 2048
            or self.reduce_source_span_bytes != 4096
            or self.reduce_destination_span_bytes != 2048
            or self.alignment_bytes != 64
            or self.scratch_roots
            != (
                S2LiteDp4ScratchRoot(0, 4096, (0, 2048)),
                S2LiteDp4ScratchRoot(2, 4096, (0, 2048)),
            )
        ):
            raise SchemaError("requires exact DP4 2x2 tree-AllReduce contract", path=path)
        for index, root in enumerate(self.scratch_roots):
            root.validate(f"{path}.scratch_roots[{index}]")


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArFlow:
    id: str
    kind: TreeArFlowKind
    source_replica_index: int
    destination_replica_index: int
    source_die_id: int
    destination_die_id: int
    gradient_bytes: int
    channel_ref: str
    send_token_ref: str
    recv_token_ref: str
    deps: tuple[str, ...]

    @classmethod
    def create(cls, *, kind: TreeArFlowKind, source: int, destination: int, deps: tuple[str, ...]) -> "S2LiteDp4TreeArFlow":
        semantic_key = {
            "kind": kind,
            "source_replica_index": source,
            "destination_replica_index": destination,
            "source_die_id": source,
            "destination_die_id": destination,
            "gradient_bytes": 2048,
            "channel_ref": f"s2_lite.dp4.channel.{kind.value}",
            "send_token_ref": f"s2_lite.dp4.send_token.{kind.value}",
            "recv_token_ref": f"s2_lite.dp4.recv_token.{kind.value}",
            "deps": deps,
        }
        return cls(
            id=stable_artifact_id(
                "s2_lite_dp4_tree_ar_flow",
                semantic_key,
                schema_version=S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        if type(self.kind) is not TreeArFlowKind or self.gradient_bytes != 2048 or not self.deps:
            raise SchemaError("flow kind/bytes/deps are not exact", path=path)
        if self.source_replica_index == self.destination_replica_index:
            raise SchemaError("flow endpoints must differ", path=path)
        expected = S2LiteDp4TreeArFlow.create(
            kind=self.kind,
            source=self.source_replica_index,
            destination=self.destination_replica_index,
            deps=self.deps,
        )
        if self != expected:
            raise SchemaError("flow endpoint/namespace/id is not canonical", path=path)


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArReduce:
    id: str
    kind: TreeArReduceKind
    executing_replica_index: int
    executing_die_id: int
    output_ref: str
    deps: tuple[str, ...]

    @classmethod
    def create(cls, *, kind: TreeArReduceKind, replica: int, deps: tuple[str, ...]) -> "S2LiteDp4TreeArReduce":
        semantic_key = {
            "kind": kind,
            "executing_replica_index": replica,
            "executing_die_id": replica,
            "output_ref": f"s2_lite.dp4.reduce_output.{kind.value}",
            "deps": deps,
        }
        return cls(
            id=stable_artifact_id(
                "s2_lite_dp4_tree_ar_reduce",
                semantic_key,
                schema_version=S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        if type(self.kind) is not TreeArReduceKind or not self.deps:
            raise SchemaError("reduce kind/deps are not exact", path=path)
        expected = S2LiteDp4TreeArReduce.create(
            kind=self.kind, replica=self.executing_replica_index, deps=self.deps
        )
        if self != expected:
            raise SchemaError("reduce endpoint/output/id is not canonical", path=path)


@dataclass(frozen=True, slots=True)
class S2LiteDp4SgdDependency:
    replica_index: int
    sgd_action_ref: str
    depends_on_step_ref: str

    def validate(self, path: str) -> None:
        if self.replica_index not in (0, 1, 2, 3) or not self.sgd_action_ref or not self.depends_on_step_ref:
            raise SchemaError("requires exact nonempty DP4 SGD dependency", path=path)


def _one_action(dag: GlobalActionDAG, marker: str) -> GlobalAction:
    matches = tuple(action for action in dag.actions if marker in getattr(action.source, "task_id", ""))
    if len(matches) != 1:
        raise SchemaError(f"requires one action matching {marker!r}", path="local_dags")
    return matches[0]


def _tree_overlay(local_dags: tuple[GlobalActionDAG, ...]) -> tuple[
    S2LiteDp4TreeArContract,
    tuple[S2LiteDp4TreeArFlow, ...],
    tuple[S2LiteDp4TreeArReduce, ...],
    tuple[S2LiteDp4SgdDependency, ...],
]:
    if len(local_dags) != 4:
        raise SchemaError("requires four local DAGs", path="local_dags")
    wgrad = tuple(_one_action(dag, ".lm_head_wgrad") for dag in local_dags)
    sgd = tuple(_one_action(dag, ".sgd_update") for dag in local_dags)
    upload10 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.UPLOAD_1_TO_0, source=1, destination=0, deps=(wgrad[1].id,))
    upload32 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.UPLOAD_3_TO_2, source=3, destination=2, deps=(wgrad[3].id,))
    reduce01 = S2LiteDp4TreeArReduce.create(kind=TreeArReduceKind.PAIR_01, replica=0, deps=(wgrad[0].id, upload10.id))
    reduce23 = S2LiteDp4TreeArReduce.create(kind=TreeArReduceKind.PAIR_23, replica=2, deps=(wgrad[2].id, upload32.id))
    partial20 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.PARTIAL_2_TO_0, source=2, destination=0, deps=(reduce23.id,))
    global0 = S2LiteDp4TreeArReduce.create(kind=TreeArReduceKind.GLOBAL_AT_0, replica=0, deps=(reduce01.id, partial20.id))
    broadcast01 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.BROADCAST_0_TO_1, source=0, destination=1, deps=(global0.id,))
    broadcast02 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.BROADCAST_0_TO_2, source=0, destination=2, deps=(global0.id,))
    broadcast23 = S2LiteDp4TreeArFlow.create(kind=TreeArFlowKind.BROADCAST_2_TO_3, source=2, destination=3, deps=(broadcast02.id,))
    contract = S2LiteDp4TreeArContract(
        S2_LITE_DP4_TREE_AR_CASE_ID, (0, 1, 2, 3), DType.FP32,
        2048, 12288, 2, 512, 2048, 4096, 2048, 64,
        (S2LiteDp4ScratchRoot(0, 4096, (0, 2048)), S2LiteDp4ScratchRoot(2, 4096, (0, 2048))),
    )
    return (
        contract,
        (upload10, upload32, partial20, broadcast01, broadcast02, broadcast23),
        (reduce01, reduce23, global0),
        (
            S2LiteDp4SgdDependency(0, sgd[0].id, global0.id),
            S2LiteDp4SgdDependency(1, sgd[1].id, broadcast01.id),
            S2LiteDp4SgdDependency(2, sgd[2].id, broadcast02.id),
            S2LiteDp4SgdDependency(3, sgd[3].id, broadcast23.id),
        ),
    )


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArGlobalAction:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteDp4TreeArSource
    scheduled: TrainScheduledIR2
    local_dags: tuple[GlobalActionDAG, ...]
    tree_contract: S2LiteDp4TreeArContract
    tree_flows: tuple[S2LiteDp4TreeArFlow, ...]
    tree_reduces: tuple[S2LiteDp4TreeArReduce, ...]
    sgd_dependencies: tuple[S2LiteDp4SgdDependency, ...]

    @classmethod
    def create(cls, *, source: S2LiteDp4TreeArSource, scheduled: TrainScheduledIR2, local_dags: tuple[GlobalActionDAG, ...]) -> "S2LiteDp4TreeArGlobalAction":
        contract, flows, reduces, dependencies = _tree_overlay(local_dags)
        semantic_key = {
            "source": source, "scheduled": scheduled, "local_dags": local_dags,
            "tree_contract": contract, "tree_flows": flows,
            "tree_reduces": reduces, "sgd_dependencies": dependencies,
        }
        result = cls(
            schema_version=S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            producer_pass="s2_lite_dp4_tree_ar_global_action",
            id=stable_artifact_id(
                "s2_lite_dp4_tree_ar_global_action",
                semantic_key,
                schema_version=S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source": self.source, "scheduled": self.scheduled,
            "local_dags": self.local_dags, "tree_contract": self.tree_contract,
            "tree_flows": self.tree_flows, "tree_reduces": self.tree_reduces,
            "sgd_dependencies": self.sgd_dependencies,
        }

    def validate(self, path: str = "s2_lite_dp4_tree_ar_global_action") -> None:
        if self.schema_version != S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp4_tree_ar_global_action":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        if type(self.source) is not S2LiteDp4TreeArSource:
            raise SchemaError("must embed typed DP4 source", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if type(self.scheduled) is not TrainScheduledIR2:
            raise SchemaError("must embed TrainScheduledIR2", path=f"{path}.scheduled")
        self.scheduled.validate(f"{path}.scheduled")
        if self.scheduled.dp_degree != 4 or len(self.scheduled.replicas) != 4 or len(self.local_dags) != 4:
            raise SchemaError("requires exactly four scheduled replicas/DAGs", path=f"{path}.scheduled")
        action_ids: set[str] = set()
        hbm_ids: set[str] = set()
        for index, (replica, dag) in enumerate(zip(self.scheduled.replicas, self.local_dags)):
            dag.validate_against(replica.projected.graph, replica.projected.projection, replica.schedule_set, f"{path}.local_dags[{index}]")
            graph = replica.projected.graph
            dies = {placement.die_id for placement in graph.groups[0].placements}
            local_actions = {action.id for action in dag.actions}
            local_hbm = {use.hbm_binding_ref for action in dag.actions for use in action.state_uses}
            if graph.source_ir0_id != self.source.graph.id or len(dag.actions) != 46 or dies != {index}:
                raise SchemaError("replica does not preserve exact 46-action die-ordered source", path=f"{path}.local_dags[{index}]")
            if action_ids & local_actions or hbm_ids & local_hbm:
                raise SchemaError("replica action/HBM namespaces must be disjoint", path=f"{path}.local_dags[{index}]")
            action_ids.update(local_actions)
            hbm_ids.update(local_hbm)
            manifest = graph.persistent_state_manifest
            assert manifest is not None
            trainable = tuple(d for d in manifest.declarations if d.identity.kind is StateKind.TRAINABLE_PARAMETER)
            frozen = tuple(d for d in manifest.declarations if d.identity.kind is StateKind.PARAMETER)
            if len(trainable) != 1 or len(frozen) != 14 or trainable[0].access is not PersistentStateAccess.READ_WRITE or any(d.access is not PersistentStateAccess.READ_ONLY for d in frozen):
                raise SchemaError("replica state rights are not Lite-exact", path=f"{path}.scheduled.replicas[{index}]")
            wgrad = _one_action(dag, ".lm_head_wgrad")
            sgd = _one_action(dag, ".sgd_update")
            if wgrad.op_kind is not OpKind.GEMM or sgd.op_kind is not OpKind.OPTIMIZER_UPDATE:
                raise SchemaError("WGRAD/SGD action kinds are not exact", path=f"{path}.local_dags[{index}]")
            wgrad_nodes = tuple(
                node for node in graph.nodes if ".lm_head_wgrad" in node.id
            )
            if len(wgrad_nodes) != 1 or len(wgrad_nodes[0].outputs) != 1:
                raise SchemaError(
                    "requires one typed LM-head WGRAD node",
                    path=f"{path}.scheduled.replicas[{index}]",
                )
            gradient = next(
                (
                    value
                    for value in graph.values
                    if value.id == wgrad_nodes[0].outputs[0]
                ),
                None,
            )
            if (
                gradient is None
                or gradient.shape != (16, 32)
                or gradient.dtype is not DType.FP32
                or 4 * 16 * 32 != self.tree_contract.gradient_bytes
            ):
                raise SchemaError(
                    "LM-head WGRAD must be exact FP32 16x32 / 2048 bytes",
                    path=f"{path}.scheduled.replicas[{index}]",
                )
        expected = _tree_overlay(self.local_dags)
        if (self.tree_contract, self.tree_flows, self.tree_reduces, self.sgd_dependencies) != expected:
            raise SchemaError("tree AR overlay is not source-exact", path=path)
        self.tree_contract.validate(f"{path}.tree_contract")
        for index, item in enumerate(self.tree_flows):
            item.validate(f"{path}.tree_flows[{index}]")
        for index, item in enumerate(self.tree_reduces):
            item.validate(f"{path}.tree_reduces[{index}]")
        for index, item in enumerate(self.sgd_dependencies):
            item.validate(f"{path}.sgd_dependencies[{index}]")
        runtime_refs = tuple(
            ref for flow in self.tree_flows
            for ref in (flow.channel_ref, flow.send_token_ref, flow.recv_token_ref)
        )
        if len(runtime_refs) != 18 or len(set(runtime_refs)) != 18 or set(runtime_refs) & action_ids:
            raise SchemaError("flow channel/token namespaces must be globally disjoint", path=f"{path}.tree_flows")
        expected_id = stable_artifact_id(
            "s2_lite_dp4_tree_ar_global_action",
            self._semantic_key(),
            schema_version=S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


__all__ = [
    "S2_LITE_DP4_TREE_AR_CASE_ID",
    "S2_LITE_DP4_TREE_AR_GLOBAL_ACTION_SCHEMA_VERSION",
    "S2_LITE_DP4_TREE_AR_SOURCE_SCHEMA_VERSION",
    "S2LiteDp4ScratchRoot",
    "S2LiteDp4SgdDependency",
    "S2LiteDp4TreeArContract",
    "S2LiteDp4TreeArFlow",
    "S2LiteDp4TreeArGlobalAction",
    "S2LiteDp4TreeArReduce",
    "S2LiteDp4TreeArSource",
    "TreeArFlowKind",
    "TreeArReduceKind",
]
