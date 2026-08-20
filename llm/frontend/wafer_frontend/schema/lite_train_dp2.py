"""Typed DP2xTP1 rooted-gradient-reduction carriers for S2-Lite."""

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


S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp2_rooted_ar_source/v1alpha1"
)
S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_dp2_rooted_ar_global_action/v1alpha1"
)


class RootedArStepKind(str, Enum):
    UPLOAD = "upload"
    ROOT_REDUCE = "root_reduce"
    DOWNLOAD = "download"


def _dp2_graph(base: S2LiteLmHeadTrainIR0) -> IR0:
    instance = base.graph.instances[0]
    replicated = replace(
        instance,
        parallel=replace(instance.parallel, dp=2),
    )
    return IR0.create(
        producer_pass="s2_lite_dp2_rooted_ar_source",
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
class S2LiteDp2RootedArSource:
    schema_version: str
    producer_pass: str
    id: str
    base: S2LiteLmHeadTrainIR0
    graph: IR0

    @classmethod
    def create(
        cls, *, base: S2LiteLmHeadTrainIR0
    ) -> "S2LiteDp2RootedArSource":
        semantic_key = {"base": base, "graph": _dp2_graph(base)}
        result = cls(
            schema_version=S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION,
            producer_pass="s2_lite_dp2_rooted_ar_source",
            id=stable_artifact_id(
                "s2_lite_dp2_rooted_ar_source",
                semantic_key,
                schema_version=S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {"base": self.base, "graph": self.graph}

    def validate(self, path: str = "s2_lite_dp2_rooted_ar_source") -> None:
        if self.schema_version != S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp2_rooted_ar_source":
            raise SchemaError(
                "must be 's2_lite_dp2_rooted_ar_source'",
                path=f"{path}.producer_pass",
            )
        if type(self.base) is not S2LiteLmHeadTrainIR0:
            raise SchemaError("must embed the exact Lite template", path=f"{path}.base")
        self.base.validate(f"{path}.base")
        self.graph.validate(f"{path}.graph")
        if self.graph != _dp2_graph(self.base):
            raise SchemaError(
                "graph must be the exact DP2 replication of the Lite template",
                path=f"{path}.graph",
            )
        instance = self.graph.instances[0]
        if (
            instance.parallel.dp != 2
            or instance.parallel.tp != 1
            or instance.parallel.pp != 1
            or instance.parallel.ep != 1
            or len(self.graph.nodes) != 29
            or len(self.graph.persistent_states) != 15
            or len(self.graph.state_accesses) != 16
        ):
            raise SchemaError(
                "requires the exact DP2xTP1 29-node Lite graph",
                path=f"{path}.graph",
            )
        expected_id = stable_artifact_id(
            "s2_lite_dp2_rooted_ar_source",
            self._semantic_key(),
            schema_version=S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class S2LiteRootedArScratchSlice:
    id: str
    replica_index: int
    offset_bytes: int
    size_bytes: int

    @classmethod
    def create(
        cls, *, replica_index: int, offset_bytes: int
    ) -> "S2LiteRootedArScratchSlice":
        semantic_key = {
            "replica_index": replica_index,
            "offset_bytes": offset_bytes,
            "size_bytes": 2048,
        }
        return cls(
            id=stable_artifact_id(
                "s2_lite_rooted_ar_scratch_slice",
                semantic_key,
                schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        for name in ("replica_index", "offset_bytes", "size_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.replica_index not in (0, 1)
            or self.offset_bytes != self.replica_index * 2048
            or self.size_bytes != 2048
        ):
            raise SchemaError("scratch slice is not rank-major exact", path=path)
        expected = S2LiteRootedArScratchSlice.create(
            replica_index=self.replica_index,
            offset_bytes=self.offset_bytes,
        )
        if self.id != expected.id:
            raise SchemaError(
                f"unstable scratch id; expected {expected.id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class S2LiteRootedArContract:
    root_replica_index: int
    root_die_id: int
    nonroot_replica_index: int
    nonroot_die_id: int
    gradient_dtype: DType
    gradient_bytes: int
    root_scratch_bytes: int
    rank_input_offsets: tuple[int, ...]
    input_stride_bytes: int
    alignment_bytes: int
    reduced_output_offset_bytes: int
    root_input_slices: tuple[S2LiteRootedArScratchSlice, ...]
    reduced_output_scratch_ref: str

    def validate(self, path: str) -> None:
        for name in (
            "root_replica_index",
            "root_die_id",
            "nonroot_replica_index",
            "nonroot_die_id",
            "gradient_bytes",
            "root_scratch_bytes",
            "input_stride_bytes",
            "alignment_bytes",
            "reduced_output_offset_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for index, offset in enumerate(self.rank_input_offsets):
            validate_uint64(offset, f"{path}.rank_input_offsets[{index}]")
        if (
            self.root_replica_index != 0
            or self.root_die_id != 0
            or self.nonroot_replica_index != 1
            or self.nonroot_die_id != 1
            or self.gradient_dtype is not DType.FP32
            or self.gradient_bytes != 2048
            or self.root_scratch_bytes != 4096
            or self.rank_input_offsets != (0, 2048)
            or self.input_stride_bytes != 2048
            or self.alignment_bytes != 64
            or self.reduced_output_offset_bytes != 0
            or self.root_input_slices
            != (
                S2LiteRootedArScratchSlice.create(
                    replica_index=0, offset_bytes=0
                ),
                S2LiteRootedArScratchSlice.create(
                    replica_index=1, offset_bytes=2048
                ),
            )
            or self.reduced_output_scratch_ref
            != self.root_input_slices[0].id
        ):
            raise SchemaError(
                "requires exact die1->die0 FP32 rooted reduction and packed 4096-byte scratch",
                path=path,
            )
        for index, scratch in enumerate(self.root_input_slices):
            scratch.validate(f"{path}.root_input_slices[{index}]")


@dataclass(frozen=True, slots=True)
class S2LiteRootedArStep:
    id: str
    kind: RootedArStepKind
    source_replica_index: int
    destination_replica_index: int
    executing_die_id: int
    gradient_bytes: int
    deps: tuple[str, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "S2LiteRootedArStep":
        return cls(
            id=stable_artifact_id(
                "s2_lite_rooted_ar_step",
                semantic_key,
                schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        if type(self.kind) is not RootedArStepKind:
            raise SchemaError("must be a RootedArStepKind", path=f"{path}.kind")
        for name in (
            "source_replica_index",
            "destination_replica_index",
            "executing_die_id",
            "gradient_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.gradient_bytes != 2048 or not self.deps:
            raise SchemaError("requires 2048 bytes and explicit deps", path=path)
        if len(set(self.deps)) != len(self.deps):
            raise SchemaError("deps must be unique", path=f"{path}.deps")
        expected_id = stable_artifact_id(
            "s2_lite_rooted_ar_step",
            {
                "kind": self.kind,
                "source_replica_index": self.source_replica_index,
                "destination_replica_index": self.destination_replica_index,
                "executing_die_id": self.executing_die_id,
                "gradient_bytes": self.gradient_bytes,
                "deps": self.deps,
            },
            schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable step id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class S2LiteSgdDependency:
    replica_index: int
    sgd_action_ref: str
    depends_on_step_ref: str

    def validate(self, path: str) -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        if self.replica_index not in (0, 1):
            raise SchemaError("replica_index must be 0 or 1", path=f"{path}.replica_index")
        if not self.sgd_action_ref or not self.depends_on_step_ref:
            raise SchemaError("action and step refs must be nonempty", path=path)


def _one_action(dag: GlobalActionDAG, marker: str) -> GlobalAction:
    matches = tuple(
        action
        for action in dag.actions
        if marker in getattr(action.source, "task_id", "")
    )
    if len(matches) != 1:
        raise SchemaError(
            f"requires one action matching {marker!r}", path="local_dags"
        )
    return matches[0]


def rooted_ar_overlay(
    local_dags: tuple[GlobalActionDAG, ...],
) -> tuple[
    S2LiteRootedArContract,
    tuple[S2LiteRootedArStep, ...],
    tuple[S2LiteSgdDependency, ...],
]:
    if len(local_dags) != 2:
        raise SchemaError("requires two local DAGs", path="local_dags")
    wgrad0 = _one_action(local_dags[0], ".lm_head_wgrad")
    wgrad1 = _one_action(local_dags[1], ".lm_head_wgrad")
    sgd0 = _one_action(local_dags[0], ".sgd_update")
    sgd1 = _one_action(local_dags[1], ".sgd_update")
    upload = S2LiteRootedArStep.create(
        kind=RootedArStepKind.UPLOAD,
        source_replica_index=1,
        destination_replica_index=0,
        executing_die_id=1,
        gradient_bytes=2048,
        deps=(wgrad1.id,),
    )
    reduce = S2LiteRootedArStep.create(
        kind=RootedArStepKind.ROOT_REDUCE,
        source_replica_index=0,
        destination_replica_index=0,
        executing_die_id=0,
        gradient_bytes=2048,
        deps=(wgrad0.id, upload.id),
    )
    download = S2LiteRootedArStep.create(
        kind=RootedArStepKind.DOWNLOAD,
        source_replica_index=0,
        destination_replica_index=1,
        executing_die_id=0,
        gradient_bytes=2048,
        deps=(reduce.id,),
    )
    return (
        S2LiteRootedArContract(
            0,
            0,
            1,
            1,
            DType.FP32,
            2048,
            4096,
            (0, 2048),
            2048,
            64,
            0,
            (
                S2LiteRootedArScratchSlice.create(
                    replica_index=0, offset_bytes=0
                ),
                S2LiteRootedArScratchSlice.create(
                    replica_index=1, offset_bytes=2048
                ),
            ),
            S2LiteRootedArScratchSlice.create(
                replica_index=0, offset_bytes=0
            ).id,
        ),
        (upload, reduce, download),
        (
            S2LiteSgdDependency(0, sgd0.id, reduce.id),
            S2LiteSgdDependency(1, sgd1.id, download.id),
        ),
    )


@dataclass(frozen=True, slots=True)
class S2LiteDp2RootedArGlobalAction:
    schema_version: str
    producer_pass: str
    id: str
    source: S2LiteDp2RootedArSource
    scheduled: TrainScheduledIR2
    local_dags: tuple[GlobalActionDAG, ...]
    ar_contract: S2LiteRootedArContract
    ar_steps: tuple[S2LiteRootedArStep, ...]
    sgd_dependencies: tuple[S2LiteSgdDependency, ...]

    @classmethod
    def create(
        cls,
        *,
        source: S2LiteDp2RootedArSource,
        scheduled: TrainScheduledIR2,
        local_dags: tuple[GlobalActionDAG, ...],
    ) -> "S2LiteDp2RootedArGlobalAction":
        contract, steps, dependencies = rooted_ar_overlay(local_dags)
        semantic_key = {
            "source": source,
            "scheduled": scheduled,
            "local_dags": local_dags,
            "ar_contract": contract,
            "ar_steps": steps,
            "sgd_dependencies": dependencies,
        }
        result = cls(
            schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            producer_pass="s2_lite_dp2_rooted_ar_global_action",
            id=stable_artifact_id(
                "s2_lite_dp2_rooted_ar_global_action",
                semantic_key,
                schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source": self.source,
            "scheduled": self.scheduled,
            "local_dags": self.local_dags,
            "ar_contract": self.ar_contract,
            "ar_steps": self.ar_steps,
            "sgd_dependencies": self.sgd_dependencies,
        }

    def validate(self, path: str = "s2_lite_dp2_rooted_ar_global_action") -> None:
        if self.schema_version != S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "s2_lite_dp2_rooted_ar_global_action":
            raise SchemaError("wrong producer pass", path=f"{path}.producer_pass")
        if type(self.source) is not S2LiteDp2RootedArSource:
            raise SchemaError("must embed typed DP2 source", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if type(self.scheduled) is not TrainScheduledIR2:
            raise SchemaError("must embed TrainScheduledIR2", path=f"{path}.scheduled")
        self.scheduled.validate(f"{path}.scheduled")
        if self.scheduled.dp_degree != 2 or len(self.scheduled.replicas) != 2:
            raise SchemaError("requires exactly two scheduled replicas", path=f"{path}.scheduled")
        if len(self.local_dags) != 2:
            raise SchemaError("requires exactly two local DAGs", path=f"{path}.local_dags")
        action_ids: set[str] = set()
        hbm_ids: set[str] = set()
        for index, (replica, dag) in enumerate(zip(self.scheduled.replicas, self.local_dags)):
            dag.validate_against(
                replica.projected.graph,
                replica.projected.projection,
                replica.schedule_set,
                f"{path}.local_dags[{index}]",
            )
            graph = replica.projected.graph
            if graph.source_ir0_id != self.source.graph.id or len(dag.actions) != 46:
                raise SchemaError("replica does not preserve the exact 46-action source", path=f"{path}.local_dags[{index}]")
            dies = {placement.die_id for placement in graph.groups[0].placements}
            if dies != {index}:
                raise SchemaError("replicas must root on die0/die1 in canonical order", path=f"{path}.scheduled.replicas[{index}]")
            local_action_ids = {action.id for action in dag.actions}
            local_hbm = {use.hbm_binding_ref for action in dag.actions for use in action.state_uses}
            if action_ids.intersection(local_action_ids) or hbm_ids.intersection(local_hbm):
                raise SchemaError("replica action/HBM namespaces must be disjoint", path=f"{path}.local_dags[{index}]")
            action_ids.update(local_action_ids)
            hbm_ids.update(local_hbm)
            manifest = graph.persistent_state_manifest
            assert manifest is not None
            trainable = tuple(d for d in manifest.declarations if d.identity.kind is StateKind.TRAINABLE_PARAMETER)
            frozen = tuple(d for d in manifest.declarations if d.identity.kind is StateKind.PARAMETER)
            if (
                len(trainable) != 1
                or len(frozen) != 14
                or trainable[0].access is not PersistentStateAccess.READ_WRITE
                or any(d.access is not PersistentStateAccess.READ_ONLY for d in frozen)
            ):
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
                or 4 * 16 * 32 != self.ar_contract.gradient_bytes
            ):
                raise SchemaError(
                    "LM-head WGRAD must be exact FP32 16x32 / 2048 bytes",
                    path=f"{path}.scheduled.replicas[{index}]",
                )
        expected = rooted_ar_overlay(self.local_dags)
        if (self.ar_contract, self.ar_steps, self.sgd_dependencies) != expected:
            raise SchemaError("rooted AR overlay is not source-exact", path=path)
        self.ar_contract.validate(f"{path}.ar_contract")
        for index, step in enumerate(self.ar_steps):
            step.validate(f"{path}.ar_steps[{index}]")
        for index, dependency in enumerate(self.sgd_dependencies):
            dependency.validate(f"{path}.sgd_dependencies[{index}]")
        expected_id = stable_artifact_id(
            "s2_lite_dp2_rooted_ar_global_action",
            self._semantic_key(),
            schema_version=S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


__all__ = [
    "RootedArStepKind",
    "S2_LITE_DP2_ROOTED_AR_GLOBAL_ACTION_SCHEMA_VERSION",
    "S2_LITE_DP2_ROOTED_AR_SOURCE_SCHEMA_VERSION",
    "S2LiteDp2RootedArGlobalAction",
    "S2LiteDp2RootedArSource",
    "S2LiteRootedArContract",
    "S2LiteRootedArScratchSlice",
    "S2LiteRootedArStep",
    "S2LiteSgdDependency",
    "rooted_ar_overlay",
]
