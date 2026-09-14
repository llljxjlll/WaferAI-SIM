"""Opt-in Dense Train v2 carrier for complete rectangular Die meshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..errors import SchemaError
from ._validation_session import mark_validation_complete, validation_seen
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import ExperimentSpec
from .ir0 import IR0, JobKind, OpPhase
from .lite_train import S2LiteTrainStage
from .persistent_state import StateKind
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest
from .train_n6 import TrainLinkedProgram


FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_train_spec/v1alpha1"
)
FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_train_plan/v1alpha1"
)
FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_train_forward_carrier/v1alpha1"
)


class FlexibleDenseTrainCoverage(str, Enum):
    FULL_FORWARD_ALL_PARAMETER_BACKWARD_SGD = (
        "full_forward_all_parameter_backward_sgd"
    )


class FlexibleDenseTrainAxis(str, Enum):
    TP = "tp"
    DP = "dp"


class FlexibleDenseTrainActionKind(str, Enum):
    PARAMETER_LOAD = "parameter_load"
    FORWARD = "forward"
    BACKWARD = "backward"
    WEIGHT_GRADIENT = "weight_gradient"
    GRADIENT_SYNC = "gradient_sync"
    SGD_UPDATE = "sgd_update"
    PARAMETER_STORE = "parameter_store"


class FlexibleDenseTrainGradientSyncRole(str, Enum):
    REDUCE_SEND = "reduce_send"
    REDUCE_RECEIVE = "reduce_receive"
    BROADCAST_SEND = "broadcast_send"
    BROADCAST_RECEIVE = "broadcast_receive"


class FlexibleDenseTrainCapabilityStatus(str, Enum):
    VERIFIED = "verified"
    FALLBACK = "fallback"
    NOT_MEASURED = "not_measured"
    OUT_OF_SCOPE = "out_of_scope"


def _sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SchemaError("must be a lowercase SHA-256", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleDenseTrainSpec:
    schema_version: str
    producer_pass: str
    id: str
    source_experiment_digest: str
    mesh: RectMeshSpec
    coverage: FlexibleDenseTrainCoverage
    lite_backward_stages: tuple[S2LiteTrainStage, ...]
    dp_degree: int
    tp_degree: int
    pp_degree: int
    micro_batch_count: int
    step_count: int
    learning_rate: float
    parameter_dtype: DType
    gradient_dtype: DType
    timing_execution: bool
    functional_execution: bool

    @classmethod
    def create(
        cls,
        *,
        source_experiment_digest: str,
        mesh: RectMeshSpec,
        learning_rate: float = 1.0e-3,
    ) -> "FlexibleDenseTrainSpec":
        semantic = {
            "source_experiment_digest": source_experiment_digest,
            "mesh": mesh,
            "coverage": FlexibleDenseTrainCoverage.FULL_FORWARD_ALL_PARAMETER_BACKWARD_SGD,
            "lite_backward_stages": (
                S2LiteTrainStage.CE_BACKWARD,
                S2LiteTrainStage.LM_HEAD_WGRAD,
                S2LiteTrainStage.SGD_UPDATE,
            ),
            "dp_degree": mesh.rows,
            "tp_degree": mesh.columns,
            "pp_degree": 1,
            "micro_batch_count": 1,
            "step_count": 1,
            "learning_rate": learning_rate,
            "parameter_dtype": DType.FP16,
            "gradient_dtype": DType.FP32,
            "timing_execution": True,
            "functional_execution": False,
        }
        result = cls(
            schema_version=FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION,
            producer_pass="flexible_dense_train_spec",
            id=stable_artifact_id(
                "flexible_dense_train_spec",
                semantic,
                schema_version=FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "flexible_dense_train_spec") -> None:
        if self.schema_version != FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "flexible_dense_train_spec":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        _sha256(self.source_experiment_digest, f"{path}.source_experiment_digest")
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        if self.coverage is not (
            FlexibleDenseTrainCoverage.FULL_FORWARD_ALL_PARAMETER_BACKWARD_SGD
        ):
            raise SchemaError("unsupported coverage", path=f"{path}.coverage")
        if self.lite_backward_stages != (
            S2LiteTrainStage.CE_BACKWARD,
            S2LiteTrainStage.LM_HEAD_WGRAD,
            S2LiteTrainStage.SGD_UPDATE,
        ):
            raise SchemaError(
                "must reuse the exact Lite CE backward -> LM-head WGRAD -> SGD stages",
                path=f"{path}.lite_backward_stages",
            )
        for name in ("dp_degree", "tp_degree", "pp_degree", "micro_batch_count", "step_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.dp_degree != self.mesh.rows
            or self.tp_degree != self.mesh.columns
            or self.dp_degree * self.tp_degree != self.mesh.rank_count
        ):
            raise SchemaError(
                "v1 requires DP=rows, TP=columns and DP*TP=R",
                path=f"{path}.dp_degree",
            )
        if (self.pp_degree, self.micro_batch_count, self.step_count) != (1, 1, 1):
            raise SchemaError(
                "v1 requires PP=1, one microbatch and one finite step", path=path
            )
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise SchemaError("must be a positive float", path=f"{path}.learning_rate")
        if self.parameter_dtype is not DType.FP16 or self.gradient_dtype is not DType.FP32:
            raise SchemaError("requires FP16 parameter and FP32 gradient", path=path)
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("v1 is timing-only", path=path)
        expected = stable_artifact_id(
            "flexible_dense_train_spec",
            self._semantic_key(),
            schema_version=FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseTrainAxisGroup:
    axis: FlexibleDenseTrainAxis
    index: int
    ranks: tuple[int, ...]

    def validate(self, path: str) -> None:
        if type(self.axis) is not FlexibleDenseTrainAxis:
            raise SchemaError("must be a FlexibleDenseTrainAxis", path=f"{path}.axis")
        validate_uint64(self.index, f"{path}.index")
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise SchemaError("must contain unique ranks", path=f"{path}.ranks")
        for index, rank in enumerate(self.ranks):
            validate_uint64(rank, f"{path}.ranks[{index}]")


@dataclass(frozen=True, slots=True)
class DenseTrainTapeBinding:
    forward_node_ref: str
    backward_node_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.forward_node_ref, f"{path}.forward_node_ref")
        validate_nonempty(self.backward_node_ref, f"{path}.backward_node_ref")


@dataclass(frozen=True, slots=True)
class DenseTrainParameterTemplate:
    """One forward parameter shard and its finite typed training expansion."""

    state_ref: str
    tensor_ref: str
    tp_shard_index: int
    owner_ranks: tuple[int, ...]
    forward_consumer_refs: tuple[str, ...]
    backward_node_refs: tuple[str, ...]
    wgrad_ref: str
    weight_bytes: int
    gradient_bytes: int

    def validate(self, path: str) -> None:
        for name in ("state_ref", "tensor_ref", "wgrad_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.tp_shard_index, f"{path}.tp_shard_index")
        for name in ("owner_ranks", "forward_consumer_refs", "backward_node_refs"):
            values = getattr(self, name)
            if type(values) is not tuple or not values or len(set(values)) != len(values):
                raise SchemaError("must be a non-empty unique tuple", path=f"{path}.{name}")
        for index, rank in enumerate(self.owner_ranks):
            validate_uint64(rank, f"{path}.owner_ranks[{index}]")
        for name in ("forward_consumer_refs", "backward_node_refs"):
            for index, ref in enumerate(getattr(self, name)):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        validate_uint64(self.weight_bytes, f"{path}.weight_bytes")
        validate_uint64(self.gradient_bytes, f"{path}.gradient_bytes")
        if self.weight_bytes == 0 or self.gradient_bytes != 2 * self.weight_bytes:
            raise SchemaError(
                "FP32 gradient bytes must be twice FP16 weight bytes", path=path
            )


@dataclass(frozen=True, slots=True)
class DenseTrainGradientTransfer:
    state_ref: str
    source_rank: int
    destination_rank: int
    logical_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        for name in ("source_rank", "destination_rank", "logical_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank or self.logical_bytes == 0:
            raise SchemaError("requires remote positive-byte transfer", path=path)


@dataclass(frozen=True, slots=True)
class DenseTrainGradientWave:
    index: int
    state_ref: str
    tp_shard_index: int
    dp_offset: int
    transfers: tuple[DenseTrainGradientTransfer, ...]
    max_sessions_per_rank: int

    def validate(self, path: str) -> None:
        validate_uint64(self.index, f"{path}.index")
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        validate_uint64(self.tp_shard_index, f"{path}.tp_shard_index")
        validate_uint64(self.dp_offset, f"{path}.dp_offset")
        if self.index == 0 or self.dp_offset == 0 or not self.transfers:
            raise SchemaError("wave index and transfers must be positive", path=path)
        if self.max_sessions_per_rank != 1:
            raise SchemaError(
                "tree wave requires one exact directed transfer",
                path=f"{path}.max_sessions_per_rank",
            )
        for index, transfer in enumerate(self.transfers):
            if type(transfer) is not DenseTrainGradientTransfer:
                raise SchemaError(
                    "must be a DenseTrainGradientTransfer",
                    path=f"{path}.transfers[{index}]",
                )
            transfer.validate(f"{path}.transfers[{index}]")
        sources = tuple(item.source_rank for item in self.transfers)
        destinations = tuple(item.destination_rank for item in self.transfers)
        if (
            sources != tuple(sorted(sources))
            or len(set(sources)) != len(sources)
            or len(set(destinations)) != len(destinations)
        ):
            raise SchemaError("must be a canonical one-send/one-receive wave", path=path)


@dataclass(frozen=True, slots=True)
class DenseTrainRankAction:
    id: str
    rank: int
    index: int
    kind: FlexibleDenseTrainActionKind
    state_ref: str | None
    op_ref: str | None
    depends_on: tuple[str, ...]
    send_peer_rank: int | None
    receive_peer_rank: int | None
    logical_bytes: int
    gradient_sync_role: FlexibleDenseTrainGradientSyncRole | None = None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.index, f"{path}.index")
        if type(self.kind) is not FlexibleDenseTrainActionKind:
            raise SchemaError("must be a FlexibleDenseTrainActionKind", path=f"{path}.kind")
        for index, dependency in enumerate(self.depends_on):
            validate_nonempty(dependency, f"{path}.depends_on[{index}]")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        if self.kind is FlexibleDenseTrainActionKind.GRADIENT_SYNC:
            if type(self.gradient_sync_role) is not FlexibleDenseTrainGradientSyncRole:
                raise SchemaError("gradient sync role is required", path=path)
            sends = self.gradient_sync_role in (
                FlexibleDenseTrainGradientSyncRole.REDUCE_SEND,
                FlexibleDenseTrainGradientSyncRole.BROADCAST_SEND,
            )
            receives = self.gradient_sync_role in (
                FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE,
                FlexibleDenseTrainGradientSyncRole.BROADCAST_RECEIVE,
            )
            if sends != (self.send_peer_rank is not None) or receives != (
                self.receive_peer_rank is not None
            ):
                raise SchemaError("peer direction does not match sync role", path=path)
        elif (
            self.gradient_sync_role is not None
            or self.send_peer_rank is not None
            or self.receive_peer_rank is not None
        ):
            raise SchemaError("sync role and peers are exact for gradient sync", path=path)
        stateful = self.kind in (
            FlexibleDenseTrainActionKind.PARAMETER_LOAD,
            FlexibleDenseTrainActionKind.WEIGHT_GRADIENT,
            FlexibleDenseTrainActionKind.GRADIENT_SYNC,
            FlexibleDenseTrainActionKind.SGD_UPDATE,
            FlexibleDenseTrainActionKind.PARAMETER_STORE,
        )
        if stateful != (self.state_ref is not None):
            raise SchemaError("state_ref is exact for parameter actions", path=path)
        if self.state_ref is not None:
            validate_nonempty(self.state_ref, f"{path}.state_ref")
        op_bound = self.kind in (
            FlexibleDenseTrainActionKind.FORWARD,
            FlexibleDenseTrainActionKind.BACKWARD,
            FlexibleDenseTrainActionKind.WEIGHT_GRADIENT,
        )
        if op_bound != (self.op_ref is not None):
            raise SchemaError("op_ref is exact for graph/WGRAD actions", path=path)
        if self.op_ref is not None:
            validate_nonempty(self.op_ref, f"{path}.op_ref")
        for name in ("send_peer_rank", "receive_peer_rank"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
                if value == self.rank:
                    raise SchemaError("peer must be remote", path=f"{path}.{name}")


def _expected_gradient_tree_edges(dp_degree: int) -> tuple[
    tuple[tuple[int, int], ...], tuple[tuple[int, int], ...],
]:
    children = tuple(range(1, dp_degree))
    reduce_edges = tuple(
        (child, (child - 1) // 2)
        for child in sorted(
            children, key=lambda item: (-((item + 1).bit_length()), -item),
        )
    )
    broadcast_edges = tuple(
        (parent, child)
        for child, parent in sorted(
            ((child, (child - 1) // 2) for child in children),
            key=lambda item: (((item[0] + 1).bit_length()), item[0]),
        )
    )
    return reduce_edges, broadcast_edges


@dataclass(frozen=True, slots=True)
class FlexibleDenseTrainPlan:
    schema_version: str
    producer_pass: str
    id: str
    spec: FlexibleDenseTrainSpec
    source_experiment: ExperimentSpec
    forward_graph: IR0
    forward_node_refs: tuple[str, ...]
    tape_bindings: tuple[DenseTrainTapeBinding, ...]
    tp_groups: tuple[DenseTrainAxisGroup, ...]
    dp_groups: tuple[DenseTrainAxisGroup, ...]
    parameter_templates: tuple[DenseTrainParameterTemplate, ...]
    lm_head_gradient_bytes_per_rank: int
    gradient_waves: tuple[DenseTrainGradientWave, ...]
    rank_actions: tuple[DenseTrainRankAction, ...]
    mesh_foundation: FlexibleDenseTrainCapabilityStatus
    forward_graph_status: FlexibleDenseTrainCapabilityStatus
    backward_carrier_status: FlexibleDenseTrainCapabilityStatus
    forward_lower_link_status: FlexibleDenseTrainCapabilityStatus
    backward_lower_link_status: FlexibleDenseTrainCapabilityStatus
    program_io_status: FlexibleDenseTrainCapabilityStatus
    runtime_status: FlexibleDenseTrainCapabilityStatus
    full_model_backward_materialized: bool

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseTrainPlan":
        result = cls(
            schema_version=FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION,
            producer_pass="flexible_dense_train_plan",
            id=stable_artifact_id(
                "flexible_dense_train_plan",
                semantic,
                schema_version=FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "flexible_dense_train_plan") -> None:
        if validation_seen(self, "flexible_dense_train_plan"):
            return
        if self.schema_version != FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "flexible_dense_train_plan":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        self.spec.validate(f"{path}.spec")
        if type(self.source_experiment) is not ExperimentSpec:
            raise SchemaError(
                "must be an ExperimentSpec", path=f"{path}.source_experiment"
            )
        self.source_experiment.validate(f"{path}.source_experiment")
        if canonical_digest(self.source_experiment) != self.spec.source_experiment_digest:
            raise SchemaError(
                "source experiment digest drifted",
                path=f"{path}.spec.source_experiment_digest",
            )
        if type(self.forward_graph) is not IR0:
            raise SchemaError("must be an IR0", path=f"{path}.forward_graph")
        self.forward_graph.validate(f"{path}.forward_graph")
        if (
            self.forward_graph.job is not JobKind.TRAIN
            or self.forward_graph.producer_pass != "train_forward_expand"
            or any(node.phase is not OpPhase.FWD for node in self.forward_graph.nodes)
        ):
            raise SchemaError("must embed exact forward-only Train IR0", path=f"{path}.forward_graph")
        from ..passes.train_forward import build_train_forward_ir0

        if self.forward_graph != build_train_forward_ir0(self.source_experiment):
            raise SchemaError(
                "forward graph must exactly derive from source experiment",
                path=f"{path}.forward_graph",
            )
        instance = self.forward_graph.instances[0]
        if (
            instance.parallel.dp != self.spec.dp_degree
            or instance.parallel.tp != self.spec.tp_degree
            or instance.parallel.pp != 1
        ):
            raise SchemaError("forward graph axis mapping disagrees with spec", path=path)
        if self.forward_node_refs != tuple(node.id for node in self.forward_graph.nodes):
            raise SchemaError("must cover every forward node in source order", path=f"{path}.forward_node_refs")
        expected_tape = tuple(
            DenseTrainTapeBinding(ref, f"backward::{ref}")
            for ref in reversed(self.forward_node_refs)
        )
        if self.tape_bindings != expected_tape:
            raise SchemaError("backward tape must exactly reverse full forward order", path=f"{path}.tape_bindings")
        expected_tp = tuple(
            DenseTrainAxisGroup(
                FlexibleDenseTrainAxis.TP,
                row,
                tuple(row * self.spec.tp_degree + col for col in range(self.spec.tp_degree)),
            )
            for row in range(self.spec.dp_degree)
        )
        expected_dp = tuple(
            DenseTrainAxisGroup(
                FlexibleDenseTrainAxis.DP,
                col,
                tuple(row * self.spec.tp_degree + col for row in range(self.spec.dp_degree)),
            )
            for col in range(self.spec.tp_degree)
        )
        if self.tp_groups != expected_tp or self.dp_groups != expected_dp:
            raise SchemaError("axis groups must be exact physical rows/columns", path=path)
        states = tuple(
            sorted(
                self.forward_graph.persistent_states,
                key=lambda item: (
                    item.identity.shard_index,
                    item.identity.tensor_ref or "",
                    item.id,
                ),
            )
        )
        if not states or any(
            state.identity.kind is not StateKind.PARAMETER
            or state.identity.tensor_ref is None
            or state.identity.shard_index >= self.spec.tp_degree
            for state in states
        ):
            raise SchemaError(
                "forward trainable templates require every exact PARAMETER shard",
                path=f"{path}.forward_graph.persistent_states",
            )
        node_order = {
            ref: index for index, ref in enumerate(self.forward_node_refs)
        }
        expected_templates = []
        for state in states:
            consumers = tuple(
                sorted(
                    {
                        access.node_ref
                        for access in self.forward_graph.state_accesses
                        if access.state_ref == state.id
                    },
                    key=node_order.__getitem__,
                )
            )
            tensor_ref = state.identity.tensor_ref
            assert tensor_ref is not None
            if not consumers:
                raise SchemaError(
                    "every parameter shard requires a forward consumer",
                    path=f"{path}.forward_graph.state_accesses",
                )
            column = state.identity.shard_index
            expected_templates.append(
                DenseTrainParameterTemplate(
                    state_ref=state.id,
                    tensor_ref=tensor_ref,
                    tp_shard_index=column,
                    owner_ranks=tuple(
                        row * self.spec.tp_degree + column
                        for row in range(self.spec.dp_degree)
                    ),
                    forward_consumer_refs=consumers,
                    backward_node_refs=tuple(
                        f"backward::{ref}::{state.id}"
                        for ref in reversed(consumers)
                    ),
                    wgrad_ref=f"wgrad::{tensor_ref}::tp{column}",
                    weight_bytes=state.tensor_bytes,
                    gradient_bytes=2 * state.tensor_bytes,
                )
            )
        expected_templates_tuple = tuple(expected_templates)
        if self.parameter_templates != expected_templates_tuple:
            raise SchemaError(
                "parameter templates must derive from all forward persistent states",
                path=f"{path}.parameter_templates",
            )
        for index, template in enumerate(self.parameter_templates):
            template.validate(f"{path}.parameter_templates[{index}]")
        validate_uint64(
            self.lm_head_gradient_bytes_per_rank,
            f"{path}.lm_head_gradient_bytes_per_rank",
        )
        lm_head_ref = f"{instance.id}.lm_head.weight"
        lm_head_templates = tuple(
            item for item in self.parameter_templates if item.tensor_ref == lm_head_ref
        )
        if (
            len(lm_head_templates) != self.spec.tp_degree
            or len({item.gradient_bytes for item in lm_head_templates}) != 1
            or self.lm_head_gradient_bytes_per_rank
            != lm_head_templates[0].gradient_bytes
        ):
            raise SchemaError(
                "LM-head compatibility bytes must derive from all TP shards",
                path=f"{path}.lm_head_gradient_bytes_per_rank",
            )
        for index, group in enumerate((*self.tp_groups, *self.dp_groups)):
            group.validate(f"{path}.groups[{index}]")
        for index, wave in enumerate(self.gradient_waves):
            wave.validate(f"{path}.gradient_waves[{index}]")
        for index, action in enumerate(self.rank_actions):
            action.validate(f"{path}.rank_actions[{index}]")
        reduce_edges, broadcast_edges = _expected_gradient_tree_edges(
            self.spec.dp_degree
        )
        expected_waves_list = []
        for template in self.parameter_templates:
            for edges in (reduce_edges, broadcast_edges):
                for source_row, destination_row in edges:
                    expected_waves_list.append(
                        DenseTrainGradientWave(
                            index=len(expected_waves_list) + 1,
                            state_ref=template.state_ref,
                            tp_shard_index=template.tp_shard_index,
                            dp_offset=len(expected_waves_list) + 1,
                            transfers=(DenseTrainGradientTransfer(
                                state_ref=template.state_ref,
                                source_rank=(source_row * self.spec.tp_degree + template.tp_shard_index),
                                destination_rank=(destination_row * self.spec.tp_degree + template.tp_shard_index),
                                logical_bytes=template.gradient_bytes,
                            ),),
                            max_sessions_per_rank=1,
                        )
                    )
        expected_waves = tuple(expected_waves_list)
        if self.gradient_waves != expected_waves:
            raise SchemaError(
                "gradient waves must sequentially cover every state and DP offset",
                path=f"{path}.gradient_waves",
            )
        by_rank_lists = {
            rank: [] for rank in range(self.spec.mesh.rank_count)
        }
        for action in self.rank_actions:
            if action.rank not in by_rank_lists:
                raise SchemaError("rank escapes Mesh", path=f"{path}.rank_actions")
            by_rank_lists[action.rank].append(action)
        by_rank = {
            rank: tuple(actions) for rank, actions in by_rank_lists.items()
        }
        for rank, actions in by_rank.items():
            row, column = divmod(rank, self.spec.tp_degree)
            local_templates = tuple(
                item
                for item in self.parameter_templates
                if item.tp_shard_index == column
            )
            expected = []
            expected.extend(
                (FlexibleDenseTrainActionKind.PARAMETER_LOAD, item.state_ref, None,
                 item.weight_bytes, None, None, None)
                for item in local_templates
            )
            expected.extend(
                (FlexibleDenseTrainActionKind.FORWARD, None, ref, 0, None, None, None)
                for ref in self.forward_node_refs
            )
            expected.extend(
                (FlexibleDenseTrainActionKind.BACKWARD, None, binding.backward_node_ref,
                 0, None, None, None)
                for binding in self.tape_bindings
            )
            expected.extend(
                (FlexibleDenseTrainActionKind.WEIGHT_GRADIENT, item.state_ref,
                 item.wgrad_ref, item.gradient_bytes, None, None, None)
                for item in local_templates
            )
            for item in local_templates:
                for child, parent in reduce_edges:
                    if row == child:
                        expected.append((
                            FlexibleDenseTrainActionKind.GRADIENT_SYNC, item.state_ref,
                            None, item.gradient_bytes, parent * self.spec.tp_degree + column,
                            None, FlexibleDenseTrainGradientSyncRole.REDUCE_SEND,
                        ))
                    elif row == parent:
                        expected.append((
                            FlexibleDenseTrainActionKind.GRADIENT_SYNC, item.state_ref,
                            None, item.gradient_bytes, None,
                            child * self.spec.tp_degree + column,
                            FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE,
                        ))
                for parent, child in broadcast_edges:
                    if row == parent:
                        expected.append((
                            FlexibleDenseTrainActionKind.GRADIENT_SYNC, item.state_ref,
                            None, item.gradient_bytes, child * self.spec.tp_degree + column,
                            None, FlexibleDenseTrainGradientSyncRole.BROADCAST_SEND,
                        ))
                    elif row == child:
                        expected.append((
                            FlexibleDenseTrainActionKind.GRADIENT_SYNC, item.state_ref,
                            None, item.gradient_bytes, None,
                            parent * self.spec.tp_degree + column,
                            FlexibleDenseTrainGradientSyncRole.BROADCAST_RECEIVE,
                        ))
            expected.extend(
                (FlexibleDenseTrainActionKind.SGD_UPDATE, item.state_ref, None,
                 item.gradient_bytes, None, None, None)
                for item in local_templates
            )
            expected.extend(
                (FlexibleDenseTrainActionKind.PARAMETER_STORE, item.state_ref, None,
                 item.weight_bytes, None, None, None)
                for item in local_templates
            )
            actual = tuple(
                (
                    item.kind, item.state_ref, item.op_ref, item.logical_bytes,
                    item.send_peer_rank, item.receive_peer_rank,
                    item.gradient_sync_role,
                )
                for item in actions
            )
            if (
                tuple(item.index for item in actions) != tuple(range(len(actions)))
                or actual != tuple(expected)
                or any(
                    action.id
                    != f"flex_train.r{rank}.s{index}.{action.kind.value}"
                    for index, action in enumerate(actions)
                )
                or any(
                    action.depends_on != (() if index == 0 else (actions[index - 1].id,))
                    for index, action in enumerate(actions)
                )
            ):
                raise SchemaError(
                    "each rank must cover all state loads/WGRAD/reductions before any SGD/store",
                    path=f"{path}.rank_actions.rank{rank}",
                )
        expected_status = FlexibleDenseTrainCapabilityStatus.VERIFIED
        if (
            self.mesh_foundation is not expected_status
            or self.forward_graph_status is not expected_status
            or self.backward_carrier_status is not expected_status
            or self.forward_lower_link_status is not FlexibleDenseTrainCapabilityStatus.NOT_MEASURED
            or self.backward_lower_link_status is not FlexibleDenseTrainCapabilityStatus.OUT_OF_SCOPE
            or self.program_io_status is not FlexibleDenseTrainCapabilityStatus.NOT_MEASURED
            or self.runtime_status is not FlexibleDenseTrainCapabilityStatus.NOT_MEASURED
            or self.full_model_backward_materialized is not False
        ):
            raise SchemaError("capability status would overclaim current evidence", path=path)
        expected = stable_artifact_id(
            "flexible_dense_train_plan",
            self._semantic_key(),
            schema_version=FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")
        mark_validation_complete(self, "flexible_dense_train_plan")


@dataclass(frozen=True, slots=True)
class FlexibleDenseTrainForwardCarrier:
    schema_version: str
    producer_pass: str
    id: str
    plan: FlexibleDenseTrainPlan
    linked_forward: TrainLinkedProgram
    core_stream_count: int
    symbolic_record_count: int

    @classmethod
    def create(
        cls,
        *,
        plan: FlexibleDenseTrainPlan,
        linked_forward: TrainLinkedProgram,
    ) -> "FlexibleDenseTrainForwardCarrier":
        semantic = {
            "plan": plan,
            "linked_forward": linked_forward,
            "core_stream_count": len(linked_forward.manifest.core_streams),
            "symbolic_record_count": sum(
                len(stream.records)
                for stream in linked_forward.manifest.core_streams
            ),
        }
        result = cls(
            schema_version=FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION,
            producer_pass="flexible_dense_train_forward_carrier",
            id=stable_artifact_id(
                "flexible_dense_train_forward_carrier",
                semantic,
                schema_version=FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "flexible_dense_train_forward_carrier") -> None:
        if validation_seen(self, "flexible_dense_train_forward_carrier"):
            return
        if self.schema_version != FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "flexible_dense_train_forward_carrier":
            raise SchemaError("requires exact producer", path=f"{path}.producer_pass")
        self.plan.validate(f"{path}.plan")
        self.linked_forward.validate(f"{path}.linked_forward")
        if self.linked_forward.source.dp_degree != self.plan.spec.dp_degree:
            raise SchemaError("linked forward DP degree drifted", path=f"{path}.linked_forward")
        if any(
            replica.lowering_context.ir1.source_ir0_id != self.plan.forward_graph.id
            for replica in self.linked_forward.source.replicas
        ):
            raise SchemaError("linked forward does not derive from plan graph", path=f"{path}.linked_forward")
        actual_cores = len(self.linked_forward.manifest.core_streams)
        actual_records = sum(
            len(stream.records)
            for stream in self.linked_forward.manifest.core_streams
        )
        if (
            self.core_stream_count != actual_cores
            or self.symbolic_record_count != actual_records
            or actual_records == 0
        ):
            raise SchemaError("forward manifest metrics do not close", path=path)
        semantic = {
            "plan": self.plan,
            "linked_forward": self.linked_forward,
            "core_stream_count": self.core_stream_count,
            "symbolic_record_count": self.symbolic_record_count,
        }
        expected = stable_artifact_id(
            "flexible_dense_train_forward_carrier",
            semantic,
            schema_version=FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")
        mark_validation_complete(self, "flexible_dense_train_forward_carrier")


__all__ = [
    "FLEXIBLE_DENSE_TRAIN_FORWARD_CARRIER_SCHEMA_VERSION",
    "FLEXIBLE_DENSE_TRAIN_PLAN_SCHEMA_VERSION",
    "FLEXIBLE_DENSE_TRAIN_SPEC_SCHEMA_VERSION",
    "DenseTrainAxisGroup",
    "DenseTrainGradientTransfer",
    "DenseTrainGradientWave",
    "DenseTrainParameterTemplate",
    "DenseTrainRankAction",
    "DenseTrainTapeBinding",
    "FlexibleDenseTrainActionKind",
    "FlexibleDenseTrainAxis",
    "FlexibleDenseTrainCapabilityStatus",
    "FlexibleDenseTrainCoverage",
    "FlexibleDenseTrainForwardCarrier",
    "FlexibleDenseTrainPlan",
    "FlexibleDenseTrainSpec",
]
