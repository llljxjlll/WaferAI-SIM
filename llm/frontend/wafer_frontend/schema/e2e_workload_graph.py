"""Typed logical coverage graph for multi-layer, multi-step E2E workloads."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import ReduceOp
from .parallel_placement import ParallelPlacement, ParallelWorkloadKind
from .workload_run import WorkloadFamily, WorkloadRunRequest


E2E_WORKLOAD_GRAPH_SCHEMA_VERSION = "wafer_frontend.e2e_workload_graph/v1alpha1"
E2E_WORKLOAD_OPERATION_SCHEMA_VERSION = (
    "wafer_frontend.e2e_workload_operation/v1alpha1"
)
E2E_STATE_VERSION_SCHEMA_VERSION = "wafer_frontend.e2e_state_version/v1alpha1"
E2E_TENSOR_VALUE_SCHEMA_VERSION = "wafer_frontend.e2e_tensor_value/v1alpha1"
E2E_ROUTE_TRACE_SCHEMA_VERSION = "wafer_frontend.e2e_route_trace/v1alpha1"


class E2EOperationKind(str, Enum):
    PARAMETER_LOAD = "parameter_load"
    EMBEDDING = "embedding"
    INPUT_NORM = "input_norm"
    QKV = "qkv"
    ROPE = "rope"
    KV_LOAD = "kv_load"
    ATTENTION = "attention"
    KV_APPEND = "kv_append"
    ATTENTION_OUT = "attention_out"
    RESIDUAL = "residual"
    POST_NORM = "post_norm"
    MLP_UP = "mlp_up"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN = "mlp_down"
    ROUTER = "router"
    ROUTE_FREEZE = "route_freeze"
    DISPATCH = "dispatch"
    EXPERT_FORWARD = "expert_forward"
    COMBINE = "combine"
    FINAL_NORM = "final_norm"
    LM_HEAD = "lm_head"
    LOGITS = "logits"
    LOSS = "loss"
    DENSE_BACKWARD = "dense_backward"
    SHARED_BACKWARD = "shared_backward"
    GRAD_DISPATCH = "grad_dispatch"
    EXPERT_BACKWARD = "expert_backward"
    DX_COMBINE = "dx_combine"
    WGRAD = "wgrad"
    ROUTER_GRADIENT = "router_gradient"
    EXPERT_GRADIENT = "expert_gradient"
    GRADIENT_SYNC = "gradient_sync"
    SGD_UPDATE = "sgd_update"
    OPTIMIZER_LOAD = "optimizer_load"
    ADAMW_UPDATE = "adamw_update"
    OPTIMIZER_STORE = "optimizer_store"
    PARAMETER_STORE = "parameter_store"
    STEP_COMMIT = "step_commit"


class E2EStateKind(str, Enum):
    PARAMETER = "parameter"
    KV = "kv"
    GRADIENT = "gradient"
    OPTIMIZER_MASTER = "optimizer_master"
    OPTIMIZER_MOMENT1 = "optimizer_moment1"
    OPTIMIZER_MOMENT2 = "optimizer_moment2"
    OPTIMIZER_STEP = "optimizer_step"
    ACTIVATION = "activation"
    LOGITS = "logits"
    LOSS = "loss"
    ROUTE_TOKEN = "route_token"
    DISPATCH_PAYLOAD = "dispatch_payload"


class E2EArtifactStatus(str, Enum):
    NOT_MATERIALIZED = "not_materialized"


@dataclass(frozen=True, slots=True)
class E2EStateVersion:
    id: str
    case_id: str
    logical_name: str
    kind: E2EStateKind
    version: int
    layer: int | None
    expert: int | None
    producer_op_id: str | None
    consumer_op_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        logical_name: str,
        kind: E2EStateKind,
        version: int,
        layer: int | None = None,
        expert: int | None = None,
        producer_op_id: str | None = None,
        consumer_op_ids: tuple[str, ...] = (),
    ) -> "E2EStateVersion":
        identity = {
            "case_id": case_id,
            "logical_name": logical_name,
            "kind": kind,
            "version": version,
            "layer": layer,
            "expert": expert,
        }
        result = cls(
            id=stable_artifact_id(
                "e2e_state_version",
                identity,
                schema_version=E2E_STATE_VERSION_SCHEMA_VERSION,
            ),
            producer_op_id=producer_op_id,
            consumer_op_ids=consumer_op_ids,
            **identity,
        )
        result.validate()
        return result

    def _identity(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "logical_name": self.logical_name,
            "kind": self.kind,
            "version": self.version,
            "layer": self.layer,
            "expert": self.expert,
        }

    def validate(self, path: str = "e2e_state_version") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        validate_nonempty(self.logical_name, f"{path}.logical_name")
        if type(self.kind) is not E2EStateKind:
            raise SchemaError("must be an E2EStateKind", path=f"{path}.kind")
        validate_uint64(self.version, f"{path}.version")
        for name in ("layer", "expert"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if self.kind is E2EStateKind.KV:
            if self.layer is None or self.expert is not None:
                raise SchemaError(
                    "KV state requires a layer and no expert", path=path
                )
        if self.kind in (E2EStateKind.ROUTE_TOKEN, E2EStateKind.DISPATCH_PAYLOAD):
            if self.layer is None:
                raise SchemaError("routing state requires a layer", path=path)
        if self.producer_op_id is not None:
            validate_nonempty(self.producer_op_id, f"{path}.producer_op_id")
        if type(self.consumer_op_ids) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.consumer_op_ids"
            )
        if len(set(self.consumer_op_ids)) != len(self.consumer_op_ids):
            raise SchemaError(
                "contains duplicate operation ids", path=f"{path}.consumer_op_ids"
            )
        for index, consumer in enumerate(self.consumer_op_ids):
            validate_nonempty(consumer, f"{path}.consumer_op_ids[{index}]")
        expected = stable_artifact_id(
            "e2e_state_version",
            self._identity(),
            schema_version=E2E_STATE_VERSION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable state version id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class E2ETensorValue:
    """Rank-local tensor view bound to one logical state version."""

    id: str
    case_id: str
    logical_name: str
    shape: tuple[int, ...]
    dtype: DType
    size_bytes: int
    state_ref: str
    owner_domain_ref: str | None
    logical_rank: int
    tp_shard: int
    producer_op_id: str | None
    consumer_op_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        logical_name: str,
        shape: tuple[int, ...],
        dtype: DType,
        state_ref: str,
        owner_domain_ref: str | None,
        logical_rank: int,
        tp_shard: int,
        producer_op_id: str | None = None,
        consumer_op_ids: tuple[str, ...] = (),
    ) -> "E2ETensorValue":
        width = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}.get(dtype)
        size_bytes = prod(shape) * width if width is not None else -1
        identity = {
            "case_id": case_id,
            "logical_name": logical_name,
            "shape": shape,
            "dtype": dtype,
            "size_bytes": size_bytes,
            "state_ref": state_ref,
            "owner_domain_ref": owner_domain_ref,
            "logical_rank": logical_rank,
            "tp_shard": tp_shard,
        }
        result = cls(
            id=stable_artifact_id(
                "e2e_tensor_value",
                identity,
                schema_version=E2E_TENSOR_VALUE_SCHEMA_VERSION,
            ),
            producer_op_id=producer_op_id,
            consumer_op_ids=consumer_op_ids,
            **identity,
        )
        result.validate()
        return result

    def _identity(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "logical_name": self.logical_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
            "state_ref": self.state_ref,
            "owner_domain_ref": self.owner_domain_ref,
            "logical_rank": self.logical_rank,
            "tp_shard": self.tp_shard,
        }

    def validate(self, path: str = "e2e_tensor_value") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        validate_nonempty(self.logical_name, f"{path}.logical_name")
        if type(self.shape) is not tuple or not self.shape:
            raise SchemaError("must be a non-empty tuple", path=f"{path}.shape")
        for index, extent in enumerate(self.shape):
            validate_uint64(extent, f"{path}.shape[{index}]")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        width = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}[self.dtype]
        if self.size_bytes != prod(self.shape) * width:
            raise SchemaError(
                "must equal product(shape) * dtype bytes",
                path=f"{path}.size_bytes",
            )
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        if self.owner_domain_ref is not None:
            validate_nonempty(self.owner_domain_ref, f"{path}.owner_domain_ref")
        validate_uint64(self.logical_rank, f"{path}.logical_rank")
        validate_uint64(self.tp_shard, f"{path}.tp_shard")
        if self.producer_op_id is not None:
            validate_nonempty(self.producer_op_id, f"{path}.producer_op_id")
        if type(self.consumer_op_ids) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.consumer_op_ids")
        if len(set(self.consumer_op_ids)) != len(self.consumer_op_ids):
            raise SchemaError("contains duplicate operation ids", path=f"{path}.consumer_op_ids")
        for index, operation_id in enumerate(self.consumer_op_ids):
            validate_nonempty(operation_id, f"{path}.consumer_op_ids[{index}]")
        expected = stable_artifact_id(
            "e2e_tensor_value",
            self._identity(),
            schema_version=E2E_TENSOR_VALUE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable tensor value id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class E2ERouteTrace:
    """Frozen top-1 token routing and its rank-local payload values."""

    id: str
    case_id: str
    phase: str
    step: int
    layer: int
    token_count: int
    expert_by_token: tuple[int, ...]
    expert_token_counts: tuple[int, ...]
    route_value_refs: tuple[str, ...]
    dispatch_value_refs: tuple[str, ...]
    expert_output_value_refs: tuple[str, ...]
    combine_value_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "E2ERouteTrace":
        result = cls(
            id=stable_artifact_id(
                "e2e_route_trace",
                semantic,
                schema_version=E2E_ROUTE_TRACE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def zero_token_experts(self) -> tuple[int, ...]:
        return tuple(
            expert
            for expert, count in enumerate(self.expert_token_counts)
            if count == 0
        )

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "e2e_route_trace") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        validate_nonempty(self.phase, f"{path}.phase")
        validate_uint64(self.step, f"{path}.step")
        validate_uint64(self.layer, f"{path}.layer")
        validate_uint64(self.token_count, f"{path}.token_count")
        if type(self.expert_by_token) is not tuple or len(self.expert_by_token) != self.token_count:
            raise SchemaError("must map every token", path=f"{path}.expert_by_token")
        if type(self.expert_token_counts) is not tuple or not self.expert_token_counts:
            raise SchemaError("must contain expert counts", path=f"{path}.expert_token_counts")
        for index, expert in enumerate(self.expert_by_token):
            if type(expert) is not int or not 0 <= expert < len(self.expert_token_counts):
                raise SchemaError("expert is out of range", path=f"{path}.expert_by_token[{index}]")
        expected_counts = tuple(
            self.expert_by_token.count(expert)
            for expert in range(len(self.expert_token_counts))
        )
        if self.expert_token_counts != expected_counts:
            raise SchemaError("counts disagree with frozen trace", path=f"{path}.expert_token_counts")
        for name in (
            "route_value_refs",
            "dispatch_value_refs",
            "expert_output_value_refs",
            "combine_value_refs",
        ):
            refs = getattr(self, name)
            if type(refs) is not tuple or not refs:
                raise SchemaError("must be a non-empty tuple", path=f"{path}.{name}")
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{name}")
        expert_count = len(self.expert_token_counts)
        if len(self.dispatch_value_refs) % expert_count != 0:
            raise SchemaError("must cover every expert", path=f"{path}.dispatch_value_refs")
        if len(self.expert_output_value_refs) != len(self.dispatch_value_refs):
            raise SchemaError("payload arity mismatch", path=f"{path}.expert_output_value_refs")
        expected = stable_artifact_id(
            "e2e_route_trace",
            self._semantic(),
            schema_version=E2E_ROUTE_TRACE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable route trace id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class E2EWorkloadOperation:
    id: str
    case_id: str
    sequence_index: int
    kind: E2EOperationKind
    phase: str
    step: int
    layer: int | None
    expert: int | None
    parameter_ref: str | None
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    input_value_refs: tuple[str, ...]
    output_value_refs: tuple[str, ...]
    group_refs: tuple[str, ...]
    reduce_op: ReduceOp | None
    normalization_denominator: int | None
    deps: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        sequence_index: int,
        kind: E2EOperationKind,
        phase: str,
        step: int,
        layer: int | None = None,
        expert: int | None = None,
        parameter_ref: str | None = None,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        input_value_refs: tuple[str, ...] = (),
        output_value_refs: tuple[str, ...] = (),
        group_refs: tuple[str, ...] = (),
        reduce_op: ReduceOp | None = None,
        normalization_denominator: int | None = None,
        deps: tuple[str, ...] = (),
    ) -> "E2EWorkloadOperation":
        semantic = {
            "case_id": case_id,
            "sequence_index": sequence_index,
            "kind": kind,
            "phase": phase,
            "step": step,
            "layer": layer,
            "expert": expert,
            "parameter_ref": parameter_ref,
            "reads": reads,
            "writes": writes,
            "input_value_refs": input_value_refs,
            "output_value_refs": output_value_refs,
            "group_refs": group_refs,
            "reduce_op": reduce_op,
            "normalization_denominator": normalization_denominator,
            "deps": deps,
        }
        result = cls(
            id=stable_artifact_id(
                "e2e_workload_operation",
                semantic,
                schema_version=E2E_WORKLOAD_OPERATION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "sequence_index": self.sequence_index,
            "kind": self.kind,
            "phase": self.phase,
            "step": self.step,
            "layer": self.layer,
            "expert": self.expert,
            "parameter_ref": self.parameter_ref,
            "reads": self.reads,
            "writes": self.writes,
            "input_value_refs": self.input_value_refs,
            "output_value_refs": self.output_value_refs,
            "group_refs": self.group_refs,
            "reduce_op": self.reduce_op,
            "normalization_denominator": self.normalization_denominator,
            "deps": self.deps,
        }

    def validate(self, path: str = "e2e_workload_operation") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        validate_uint64(self.sequence_index, f"{path}.sequence_index")
        if type(self.kind) is not E2EOperationKind:
            raise SchemaError("must be an E2EOperationKind", path=f"{path}.kind")
        validate_nonempty(self.phase, f"{path}.phase")
        validate_uint64(self.step, f"{path}.step")
        for name in ("layer", "expert"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        if self.parameter_ref is not None:
            validate_nonempty(self.parameter_ref, f"{path}.parameter_ref")
        for name in (
            "reads",
            "writes",
            "input_value_refs",
            "output_value_refs",
            "group_refs",
            "deps",
        ):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise SchemaError(
                    "must be an immutable tuple", path=f"{path}.{name}"
                )
            if len(set(values)) != len(values):
                raise SchemaError("contains duplicate ids", path=f"{path}.{name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if set(self.reads).intersection(self.writes):
            raise SchemaError("cannot read and write one state version", path=path)
        if set(self.input_value_refs).intersection(self.output_value_refs):
            raise SchemaError("cannot read and write one tensor value", path=path)
        if self.kind is E2EOperationKind.GRADIENT_SYNC:
            if not self.group_refs or self.reduce_op is not ReduceOp.SUM:
                raise SchemaError(
                    "gradient sync requires P1 group refs and SUM reduction",
                    path=path,
                )
            if (
                type(self.normalization_denominator) is not int
                or self.normalization_denominator <= 0
            ):
                raise SchemaError(
                    "gradient sync requires a positive normalization denominator",
                    path=f"{path}.normalization_denominator",
                )
        elif (
            self.group_refs
            or self.reduce_op is not None
            or self.normalization_denominator is not None
        ):
            raise SchemaError(
                "only gradient sync carries reduction metadata", path=path
            )
        expected = stable_artifact_id(
            "e2e_workload_operation",
            self._semantic(),
            schema_version=E2E_WORKLOAD_OPERATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable operation id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class E2EWorkloadGraph:
    schema_version: str
    producer_pass: str
    id: str
    request: WorkloadRunRequest
    family: WorkloadFamily
    placement: ParallelPlacement
    operations: tuple[E2EWorkloadOperation, ...]
    state_versions: tuple[E2EStateVersion, ...]
    tensor_values: tuple[E2ETensorValue, ...]
    route_traces: tuple[E2ERouteTrace, ...]
    lowering_status: E2EArtifactStatus
    runtime_status: E2EArtifactStatus

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        placement: ParallelPlacement,
        operations: tuple[E2EWorkloadOperation, ...],
        state_versions: tuple[E2EStateVersion, ...],
        tensor_values: tuple[E2ETensorValue, ...],
        route_traces: tuple[E2ERouteTrace, ...],
    ) -> "E2EWorkloadGraph":
        semantic = {
            "request": request,
            "family": request.family,
            "placement": placement,
            "operations": operations,
            "state_versions": state_versions,
            "tensor_values": tensor_values,
            "route_traces": route_traces,
            "lowering_status": E2EArtifactStatus.NOT_MATERIALIZED,
            "runtime_status": E2EArtifactStatus.NOT_MATERIALIZED,
        }
        result = cls(
            schema_version=E2E_WORKLOAD_GRAPH_SCHEMA_VERSION,
            producer_pass="build_e2e_workload_graph",
            id=stable_artifact_id(
                "e2e_workload_graph",
                semantic,
                schema_version=E2E_WORKLOAD_GRAPH_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            "request": self.request,
            "family": self.family,
            "placement": self.placement,
            "operations": self.operations,
            "state_versions": self.state_versions,
            "tensor_values": self.tensor_values,
            "route_traces": self.route_traces,
            "lowering_status": self.lowering_status,
            "runtime_status": self.runtime_status,
        }

    def validate(self, path: str = "e2e_workload_graph") -> None:
        if self.schema_version != E2E_WORKLOAD_GRAPH_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "build_e2e_workload_graph":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        if type(self.request) is not WorkloadRunRequest:
            raise SchemaError("must be a WorkloadRunRequest", path=f"{path}.request")
        self.request.validate(f"{path}.request")
        if self.family is not self.request.family:
            raise SchemaError("family differs from request", path=f"{path}.family")
        if type(self.placement) is not ParallelPlacement:
            raise SchemaError("must be a ParallelPlacement", path=f"{path}.placement")
        self.placement.validate(f"{path}.placement")
        expected_workload_kind = (
            ParallelWorkloadKind.MOE
            if self.request.family.is_moe
            else ParallelWorkloadKind.DENSE
        )
        if (
            self.placement.workload_kind is not expected_workload_kind
            or self.placement.mesh.rows != self.request.mesh.rows
            or self.placement.mesh.columns != self.request.mesh.columns
            or self.placement.tp_degree != self.request.parallel.tp
            or self.placement.dp_degree != self.request.parallel.dp
            or self.placement.ep_degree != self.request.parallel.ep
            or self.placement.pp_degree != self.request.parallel.pp
        ):
            raise SchemaError(
                "placement does not match workload request", path=f"{path}.placement"
            )
        if (
            self.lowering_status is not E2EArtifactStatus.NOT_MATERIALIZED
            or self.runtime_status is not E2EArtifactStatus.NOT_MATERIALIZED
        ):
            raise SchemaError(
                "logical graph cannot claim lowering or runtime completion", path=path
            )
        op_by_id: dict[str, E2EWorkloadOperation] = {}
        for index, operation in enumerate(self.operations):
            op_path = f"{path}.operations[{index}]"
            if type(operation) is not E2EWorkloadOperation:
                raise SchemaError("must be an E2EWorkloadOperation", path=op_path)
            operation.validate(op_path)
            if operation.case_id != self.request.case_id:
                raise SchemaError("operation belongs to another case", path=op_path)
            if operation.sequence_index != index:
                raise SchemaError(
                    "sequence indices must be contiguous",
                    path=f"{op_path}.sequence_index",
                )
            if operation.id in op_by_id:
                raise SchemaError("duplicate operation id", path=f"{op_path}.id")
            for dependency in operation.deps:
                if dependency not in op_by_id:
                    raise SchemaError(
                        "dependency must reference an earlier operation",
                        path=f"{op_path}.deps",
                    )
            op_by_id[operation.id] = operation
        state_by_id: dict[str, E2EStateVersion] = {}
        for index, state in enumerate(self.state_versions):
            state_path = f"{path}.state_versions[{index}]"
            if type(state) is not E2EStateVersion:
                raise SchemaError("must be an E2EStateVersion", path=state_path)
            state.validate(state_path)
            if state.case_id != self.request.case_id:
                raise SchemaError("state belongs to another case", path=state_path)
            if state.id in state_by_id:
                raise SchemaError("duplicate state version id", path=f"{state_path}.id")
            state_by_id[state.id] = state
        owner_by_id = {item.id: item for item in self.placement.ownership_domains}
        rank_by_id = {
            item.logical_rank: item for item in self.placement.rank_placements
        }
        value_by_id: dict[str, E2ETensorValue] = {}
        for index, value in enumerate(self.tensor_values):
            value_path = f"{path}.tensor_values[{index}]"
            if type(value) is not E2ETensorValue:
                raise SchemaError("must be an E2ETensorValue", path=value_path)
            value.validate(value_path)
            if value.id in value_by_id:
                raise SchemaError("duplicate tensor value id", path=f"{value_path}.id")
            if value.case_id != self.request.case_id:
                raise SchemaError("value belongs to another case", path=value_path)
            if value.state_ref not in state_by_id:
                raise SchemaError("value references unknown state", path=f"{value_path}.state_ref")
            placement = rank_by_id.get(value.logical_rank)
            if placement is None or placement.coordinate.tp != value.tp_shard:
                raise SchemaError("value rank/shard is not placed", path=value_path)
            if value.owner_domain_ref is not None:
                owner = owner_by_id.get(value.owner_domain_ref)
                if (
                    owner is None
                    or value.logical_rank not in owner.replica_ranks
                    or owner.tp_shard != value.tp_shard
                ):
                    raise SchemaError("value owner/rank/shard disagree", path=value_path)
            value_by_id[value.id] = value
        if not value_by_id:
            raise SchemaError("must contain tensor values", path=f"{path}.tensor_values")
        group_by_id = {item.id: item for item in self.placement.groups}
        value_writers: dict[str, str] = {}
        value_consumers: dict[str, list[str]] = {
            value_id: [] for value_id in value_by_id
        }
        writers: dict[str, str] = {}
        consumers: dict[str, list[str]] = {state_id: [] for state_id in state_by_id}
        for operation in self.operations:
            for group_ref in operation.group_refs:
                group = group_by_id.get(group_ref)
                if group is None:
                    raise SchemaError("operation references unknown P1 group", path=f"{path}.operations")
                if operation.normalization_denominator != len(group.ranks):
                    raise SchemaError("gradient normalization must equal group size", path=f"{path}.operations")
            for value_id in operation.input_value_refs:
                if value_id not in value_by_id:
                    raise SchemaError("operation reads unknown value", path=f"{path}.operations")
                value_consumers[value_id].append(operation.id)
            for value_id in operation.output_value_refs:
                if value_id not in value_by_id:
                    raise SchemaError("operation writes unknown value", path=f"{path}.operations")
                if value_id in value_writers:
                    raise SchemaError("tensor value has multiple producers", path=f"{path}.operations")
                value_writers[value_id] = operation.id
            for state_id in operation.reads:
                if state_id not in state_by_id:
                    raise SchemaError(
                        "operation reads an unknown state", path=f"{path}.operations"
                    )
                consumers[state_id].append(operation.id)
            for state_id in operation.writes:
                if state_id not in state_by_id:
                    raise SchemaError(
                        "operation writes an unknown state", path=f"{path}.operations"
                    )
                if state_id in writers:
                    raise SchemaError(
                        "state version has multiple producers",
                        path=f"{path}.operations",
                    )
                writers[state_id] = operation.id
        for value_id, value in value_by_id.items():
            if value.producer_op_id != value_writers.get(value_id):
                raise SchemaError("value producer lineage is incomplete", path=f"{path}.tensor_values")
            if value.consumer_op_ids != tuple(value_consumers[value_id]):
                raise SchemaError("value consumer lineage is incomplete", path=f"{path}.tensor_values")
            if value.producer_op_id is not None and value.producer_op_id not in op_by_id:
                raise SchemaError("value producer is absent", path=f"{path}.tensor_values")
        for state_id, state in state_by_id.items():
            if state.producer_op_id != writers.get(state_id):
                raise SchemaError(
                    "state producer lineage disagrees with operation writes",
                    path=f"{path}.state_versions",
                )
            if state.consumer_op_ids != tuple(consumers[state_id]):
                raise SchemaError(
                    "state consumer lineage disagrees with operation reads",
                    path=f"{path}.state_versions",
                )
            if (
                state.producer_op_id is not None
                and state.producer_op_id not in op_by_id
            ):
                raise SchemaError(
                    "state producer is absent", path=f"{path}.state_versions"
                )
        seen_traces: set[tuple[str, int, int]] = set()
        for index, trace in enumerate(self.route_traces):
            trace_path = f"{path}.route_traces[{index}]"
            if type(trace) is not E2ERouteTrace:
                raise SchemaError("must be an E2ERouteTrace", path=trace_path)
            trace.validate(trace_path)
            if trace.case_id != self.request.case_id:
                raise SchemaError("trace belongs to another case", path=trace_path)
            key = (trace.phase, trace.step, trace.layer)
            if key in seen_traces:
                raise SchemaError("duplicate route trace", path=trace_path)
            seen_traces.add(key)
            for value_ref in (
                *trace.route_value_refs,
                *trace.dispatch_value_refs,
                *trace.expert_output_value_refs,
                *trace.combine_value_refs,
            ):
                if value_ref not in value_by_id:
                    raise SchemaError("trace references unknown value", path=trace_path)
        if self.request.family.is_moe != bool(self.route_traces):
            raise SchemaError("route traces must match workload family", path=f"{path}.route_traces")
        expected_id = stable_artifact_id(
            "e2e_workload_graph",
            self._semantic(),
            schema_version=E2E_WORKLOAD_GRAPH_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable workload graph id", path=f"{path}.id")


__all__ = [
    "E2EArtifactStatus",
    "E2EOperationKind",
    "E2ERouteTrace",
    "E2E_STATE_VERSION_SCHEMA_VERSION",
    "E2E_ROUTE_TRACE_SCHEMA_VERSION",
    "E2E_TENSOR_VALUE_SCHEMA_VERSION",
    "E2E_WORKLOAD_GRAPH_SCHEMA_VERSION",
    "E2E_WORKLOAD_OPERATION_SCHEMA_VERSION",
    "E2EStateKind",
    "E2EStateVersion",
    "E2ETensorValue",
    "E2EWorkloadGraph",
    "E2EWorkloadOperation",
]
