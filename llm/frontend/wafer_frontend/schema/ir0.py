"""Versioned logical graph (IR-0), with no physical placement fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
from enum import Enum

from ..errors import SchemaError, UnsupportedFeatureError
from .common import (
    DType,
    MeshAxisName,
    ProfileKey,
    TensorValue,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
)
from .stage3_profile import Stage3ProfileMode, Stage3StaticProfile


IR0_SCHEMA_VERSION = "wafer_frontend.ir0/v1alpha11"
STATE_ACCESS_SCHEMA_VERSION = "wafer_frontend.state_access/v1alpha2"


class JobKind(str, Enum):
    INFER = "infer"
    TRAIN = "train"


class LogicalRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"
    TRAIN = "train"


class OpKind(str, Enum):
    GEMM = "gemm"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    ATTENTION = "attention"
    COLLECTIVE = "collective"
    P2P = "p2p"
    EMBEDDING = "embedding"
    ROPE = "rope"
    SAMPLING = "sampling"
    CE_FORWARD = "ce_forward"
    CE_BACKWARD = "ce_backward"
    OPTIMIZER_UPDATE = "optimizer_update"


class OpPhase(str, Enum):
    FWD = "fwd"
    DGRAD = "dgrad"
    WGRAD = "wgrad"
    RECOMPUTE = "recompute"
    UPDATE = "update"


class EdgeKind(str, Enum):
    DATA = "data"
    CONTROL = "control"


class EffectKind(str, Enum):
    PURE = "pure"
    RNG = "rng"
    STATEFUL = "stateful"
    INPLACE = "inplace"


class StateAccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True, slots=True)
class StateAccess:
    """One logical state access by one SPMD rank of an IR-0 node.

    A null direction view means the whole declaration.  Explicit views are
    dense row-major rectangular slices; READ_WRITE may use different read and
    write views so decode can load context while appending only query tokens.
    """

    id: str
    node_ref: str
    state_ref: str
    mode: StateAccessMode
    rank: int
    read_offset: tuple[int, ...] | None = None
    read_shape: tuple[int, ...] | None = None
    write_offset: tuple[int, ...] | None = None
    write_shape: tuple[int, ...] | None = None

    @classmethod
    def create(
        cls,
        *,
        node_ref: str,
        state_ref: str,
        mode: StateAccessMode,
        rank: int,
        read_offset: tuple[int, ...] | None = None,
        read_shape: tuple[int, ...] | None = None,
        write_offset: tuple[int, ...] | None = None,
        write_shape: tuple[int, ...] | None = None,
    ) -> "StateAccess":
        key = {
            "node_ref": node_ref,
            "state_ref": state_ref,
            "mode": mode,
            "rank": rank,
            "read_offset": read_offset,
            "read_shape": read_shape,
            "write_offset": write_offset,
            "write_shape": write_shape,
        }
        result = cls(
            id=stable_artifact_id(
                "state_access", key, schema_version=STATE_ACCESS_SCHEMA_VERSION
            ),
            **key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "node_ref": self.node_ref,
            "state_ref": self.state_ref,
            "mode": self.mode,
            "rank": self.rank,
            "read_offset": self.read_offset,
            "read_shape": self.read_shape,
            "write_offset": self.write_offset,
            "write_shape": self.write_shape,
        }

    def validate(self, path: str = "state_access") -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        if type(self.mode) is not StateAccessMode:
            raise SchemaError("must be a StateAccessMode", path=f"{path}.mode")
        validate_uint64(self.rank, f"{path}.rank")
        active = {
            "read": self.mode in (StateAccessMode.READ, StateAccessMode.READ_WRITE),
            "write": self.mode in (StateAccessMode.WRITE, StateAccessMode.READ_WRITE),
        }
        for direction in ("read", "write"):
            offset = getattr(self, f"{direction}_offset")
            shape = getattr(self, f"{direction}_shape")
            if not active[direction] and (offset is not None or shape is not None):
                raise SchemaError(
                    "inactive direction must not define a tensor view",
                    path=f"{path}.{direction}_offset",
                )
            if (offset is None) != (shape is None):
                raise SchemaError(
                    "offset and shape must both be null or both be tuples",
                    path=f"{path}.{direction}_offset",
                )
            if offset is None:
                continue
            if type(offset) is not tuple or type(shape) is not tuple or not offset:
                raise SchemaError(
                    "must be non-empty immutable tuples",
                    path=f"{path}.{direction}_offset",
                )
            if len(offset) != len(shape):
                raise SchemaError(
                    "offset and shape must have equal rank",
                    path=f"{path}.{direction}_shape",
                )
            for index, (start, extent) in enumerate(zip(offset, shape)):
                validate_uint64(start, f"{path}.{direction}_offset[{index}]")
                validate_uint64(extent, f"{path}.{direction}_shape[{index}]")
                if extent == 0:
                    raise SchemaError(
                        "must be greater than zero",
                        path=f"{path}.{direction}_shape[{index}]",
                    )
        expected = stable_artifact_id(
            "state_access",
            self._semantic_key(),
            schema_version=STATE_ACCESS_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


def state_access_tensor_view(
    access: StateAccess,
    declaration: PersistentStateDecl,
    direction: str,
    *,
    path: str = "state_access",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return and validate one direction's dense view in declaration space."""

    if direction not in ("read", "write"):
        raise SchemaError("must be 'read' or 'write'", path=f"{path}.direction")
    enabled = {
        "read": access.mode in (StateAccessMode.READ, StateAccessMode.READ_WRITE),
        "write": access.mode in (StateAccessMode.WRITE, StateAccessMode.READ_WRITE),
    }[direction]
    if not enabled:
        raise SchemaError("direction is not active", path=f"{path}.mode")
    offset = getattr(access, f"{direction}_offset")
    shape = getattr(access, f"{direction}_shape")
    if offset is None:
        return (0,) * len(declaration.shape), declaration.shape
    assert shape is not None
    if len(offset) != len(declaration.shape):
        raise SchemaError(
            "view rank must equal the state declaration rank",
            path=f"{path}.{direction}_offset",
        )
    for index, (start, extent, bound) in enumerate(
        zip(offset, shape, declaration.shape)
    ):
        if start > bound or extent > bound - start:
            raise SchemaError(
                "view must be contained in the state declaration",
                path=f"{path}.{direction}_shape[{index}]",
            )
    return offset, shape

class CollectiveKind(str, Enum):
    ALL_GATHER = "all_gather"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"


class ReduceOp(str, Enum):
    SUM = "sum"
    MAX = "max"


class CollectiveRole(str, Enum):
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    WEIGHT_GATHER = "weight_gather"


class FusionImpl(str, Enum):
    NONE = "none"
    NAIVE = "naive"
    SWIZZLE_TOPO = "swizzle_topo"


class FusionOrigin(str, Enum):
    DECLARED = "declared"
    DISCOVERED = "discovered"


class NumericalPolicy(str, Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


class PipelineSchedule(str, Enum):
    GPIPE = "gpipe"
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED_ONE_F_ONE_B = "interleaved_1f1b"


class RecomputeMode(str, Enum):
    NONE = "none"
    SELECTIVE = "selective"
    FULL = "full"


class GemmPartition(str, Enum):
    REPLICATED = "replicated"
    COLUMN_PARALLEL = "column_parallel"
    ROW_PARALLEL = "row_parallel"
    SEQUENCE_PARALLEL_REPLICATED_WEIGHT = "sequence_parallel_replicated_weight"


class AttentionMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"
    TRAIN_FORWARD = "train_forward"


class EmbeddingTablePlacement(str, Enum):
    REPLICATED = "replicated"


class PackedQkvLayout(str, Enum):
    Q_K_V = "q_k_v"


class SamplingMode(str, Enum):
    GREEDY = "greedy"


class SampleRowSelection(str, Enum):
    LAST_PER_SEQUENCE = "last_per_sequence"


class CrossEntropyReduction(str, Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


@dataclass(frozen=True, slots=True)
class MeshAxis:
    name: MeshAxisName
    size: int

    def validate(self, path: str) -> None:
        validate_uint64(self.size, f"{path}.size")
        if self.size == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.size")


@dataclass(frozen=True, slots=True)
class DeviceMesh:
    id: str
    axes: tuple[MeshAxis, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        if not self.axes:
            raise SchemaError("must contain at least one axis", path=f"{path}.axes")
        names: set[MeshAxisName] = set()
        for index, axis in enumerate(self.axes):
            axis.validate(f"{path}.axes[{index}]")
            if axis.name in names:
                raise SchemaError(
                    f"duplicate axis {axis.name.value!r}", path=f"{path}.axes[{index}].name"
                )
            names.add(axis.name)


@dataclass(frozen=True, slots=True)
class ParallelAxes:
    tp: int
    sp: bool
    dp: int
    pp: int
    ep: int

    def validate(self, path: str) -> None:
        for name in ("tp", "dp", "pp", "ep"):
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class LogicalInstance:
    id: str
    role: LogicalRole
    replicas: int
    parallel: ParallelAxes
    meshes: tuple[DeviceMesh, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.replicas, f"{path}.replicas")
        if self.replicas == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.replicas")
        self.parallel.validate(f"{path}.parallel")
        mesh_ids: set[str] = set()
        for index, mesh in enumerate(self.meshes):
            mesh.validate(f"{path}.meshes[{index}]")
            if mesh.id in mesh_ids:
                raise SchemaError(
                    f"duplicate id {mesh.id!r}", path=f"{path}.meshes[{index}].id"
                )
            mesh_ids.add(mesh.id)


@dataclass(frozen=True, slots=True)
class GemmWorkload:
    """Logical and symmetric rank-local GEMM shapes in ``(M, N, K)`` order."""

    logical_shape: tuple[int, int, int]
    rank_shape: tuple[int, int, int]
    partition: GemmPartition
    dtype: DType

    def validate(self, path: str) -> None:
        _validate_shape(self.logical_shape, f"{path}.logical_shape", expected_rank=3)
        _validate_shape(self.rank_shape, f"{path}.rank_shape", expected_rank=3)
        if type(self.partition) is not GemmPartition:
            raise SchemaError("must be a GemmPartition", path=f"{path}.partition")
        _validate_legacy_dtype(self.dtype, f"{path}.dtype")
        logical_m, logical_n, logical_k = self.logical_shape
        rank_m, rank_n, rank_k = self.rank_shape
        if self.partition is GemmPartition.REPLICATED:
            if self.rank_shape != self.logical_shape:
                raise SchemaError(
                    "replicated GEMM requires rank_shape == logical_shape",
                    path=f"{path}.rank_shape",
                )
        elif self.partition is GemmPartition.COLUMN_PARALLEL:
            if (rank_m, rank_k) != (logical_m, logical_k):
                raise SchemaError(
                    "column-parallel GEMM may shard only N",
                    path=f"{path}.rank_shape",
                )
            _validate_partitioned_extent(logical_n, rank_n, f"{path}.rank_shape[1]")
        elif self.partition is GemmPartition.ROW_PARALLEL:
            if (rank_m, rank_n) != (logical_m, logical_n):
                raise SchemaError(
                    "row-parallel GEMM may shard only K",
                    path=f"{path}.rank_shape",
                )
            _validate_partitioned_extent(logical_k, rank_k, f"{path}.rank_shape[2]")
        else:
            if (rank_n, rank_k) != (logical_n, logical_k):
                raise SchemaError(
                    "sequence-parallel replicated-weight GEMM may shard only M",
                    path=f"{path}.rank_shape",
                )
            _validate_partitioned_extent(logical_m, rank_m, f"{path}.rank_shape[0]")


def _validate_dtype(dtype: DType, path: str) -> None:
    if type(dtype) is not DType:
        raise SchemaError("must be a DType", path=path)


def _validate_legacy_dtype(dtype: DType, path: str) -> None:
    _validate_dtype(dtype, path)
    if dtype not in (DType.FP16, DType.FP32):
        raise SchemaError("current workload requires FP16 or FP32", path=path)


def _validate_shape(
    shape: tuple[int, ...], path: str, *, expected_rank: int | None = None
) -> None:
    if type(shape) is not tuple or not shape:
        raise SchemaError("must be a non-empty immutable shape", path=path)
    if expected_rank is not None and len(shape) != expected_rank:
        raise SchemaError(f"must have rank {expected_rank}", path=path)
    for index, dimension in enumerate(shape):
        validate_uint64(dimension, f"{path}[{index}]")
        if dimension == 0:
            raise SchemaError("must be greater than zero", path=f"{path}[{index}]")


def _validate_partitioned_extent(logical: int, rank: int, path: str) -> None:
    if rank > logical or logical % rank != 0:
        raise SchemaError(
            "rank-local extent must not exceed and must exactly divide the logical extent",
            path=path,
        )


def _validate_rank_shape(
    logical: tuple[int, ...], rank: tuple[int, ...], path: str
) -> None:
    if len(logical) != len(rank):
        raise SchemaError("rank-local tensor rank must equal logical tensor rank", path=path)
    for axis, (logical_extent, rank_extent) in enumerate(zip(logical, rank)):
        _validate_partitioned_extent(logical_extent, rank_extent, f"{path}[{axis}]")


@dataclass(frozen=True, slots=True)
class NormWorkload:
    logical_input_shape: tuple[int, ...]
    logical_output_shape: tuple[int, ...]
    rank_input_shape: tuple[int, ...]
    rank_output_shape: tuple[int, ...]
    dtype: DType

    def validate(self, path: str) -> None:
        for field_name in (
            "logical_input_shape",
            "logical_output_shape",
            "rank_input_shape",
            "rank_output_shape",
        ):
            _validate_shape(getattr(self, field_name), f"{path}.{field_name}")
        if self.logical_input_shape != self.logical_output_shape:
            raise SchemaError("normalization must preserve logical shape", path=f"{path}.logical_output_shape")
        if self.rank_input_shape != self.rank_output_shape:
            raise SchemaError("normalization must preserve rank-local shape", path=f"{path}.rank_output_shape")
        _validate_rank_shape(self.logical_input_shape, self.rank_input_shape, f"{path}.rank_input_shape")
        _validate_legacy_dtype(self.dtype, f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class ElementwiseWorkload:
    logical_input_shapes: tuple[tuple[int, ...], ...]
    logical_output_shape: tuple[int, ...]
    rank_input_shapes: tuple[tuple[int, ...], ...]
    rank_output_shape: tuple[int, ...]
    dtype: DType

    def validate(self, path: str) -> None:
        if not self.logical_input_shapes:
            raise SchemaError("must contain at least one input shape", path=f"{path}.logical_input_shapes")
        if len(self.rank_input_shapes) != len(self.logical_input_shapes):
            raise SchemaError("must contain one rank-local shape per logical input", path=f"{path}.rank_input_shapes")
        _validate_shape(self.logical_output_shape, f"{path}.logical_output_shape")
        _validate_shape(self.rank_output_shape, f"{path}.rank_output_shape")
        _validate_rank_shape(self.logical_output_shape, self.rank_output_shape, f"{path}.rank_output_shape")
        output_rank = len(self.logical_output_shape)
        for index, (logical, rank) in enumerate(zip(self.logical_input_shapes, self.rank_input_shapes)):
            _validate_shape(logical, f"{path}.logical_input_shapes[{index}]", expected_rank=output_rank)
            _validate_shape(rank, f"{path}.rank_input_shapes[{index}]", expected_rank=output_rank)
            _validate_rank_shape(logical, rank, f"{path}.rank_input_shapes[{index}]")
        _validate_legacy_dtype(self.dtype, f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class RmsNormWorkload:
    logical_activation_shape: tuple[int, int]
    logical_output_shape: tuple[int, int]
    rank_activation_shape: tuple[int, int]
    rank_output_shape: tuple[int, int]
    logical_weight_shape: tuple[int]
    rank_weight_shape: tuple[int]
    epsilon: float
    dtype: DType

    def validate(self, path: str) -> None:
        for field_name, expected_rank in (
            ("logical_activation_shape", 2),
            ("logical_output_shape", 2),
            ("rank_activation_shape", 2),
            ("rank_output_shape", 2),
            ("logical_weight_shape", 1),
            ("rank_weight_shape", 1),
        ):
            _validate_shape(
                getattr(self, field_name),
                f"{path}.{field_name}",
                expected_rank=expected_rank,
            )
        if self.logical_output_shape != self.logical_activation_shape:
            raise SchemaError(
                "RMSNorm must preserve logical activation shape",
                path=f"{path}.logical_output_shape",
            )
        if self.rank_output_shape != self.rank_activation_shape:
            raise SchemaError(
                "RMSNorm must preserve rank-local activation shape",
                path=f"{path}.rank_output_shape",
            )
        _validate_rank_shape(
            self.logical_activation_shape,
            self.rank_activation_shape,
            f"{path}.rank_activation_shape",
        )
        expected_weight = (self.logical_activation_shape[-1],)
        if self.logical_weight_shape != expected_weight:
            raise SchemaError(
                f"must equal hidden shape {expected_weight!r}",
                path=f"{path}.logical_weight_shape",
            )
        if self.rank_weight_shape != self.logical_weight_shape:
            raise SchemaError(
                "RMSNorm scale must be replicated on every rank",
                path=f"{path}.rank_weight_shape",
            )
        if type(self.epsilon) is not float or not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.epsilon")
        if self.dtype is not DType.FP16:
            raise SchemaError("current RMSNorm requires FP16", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class SwiGluWorkload:
    logical_input_shape: tuple[int, int]
    logical_output_shape: tuple[int, int]
    rank_input_shape: tuple[int, int]
    rank_output_shape: tuple[int, int]
    dtype: DType

    def validate(self, path: str) -> None:
        for field_name in (
            "logical_input_shape",
            "logical_output_shape",
            "rank_input_shape",
            "rank_output_shape",
        ):
            _validate_shape(getattr(self, field_name), f"{path}.{field_name}", expected_rank=2)
        logical_m, logical_two_i = self.logical_input_shape
        rank_m, rank_two_i = self.rank_input_shape
        if self.logical_output_shape != (logical_m, logical_two_i // 2) or logical_two_i % 2:
            raise SchemaError("SwiGLU output must halve the logical feature axis", path=f"{path}.logical_output_shape")
        if self.rank_output_shape != (rank_m, rank_two_i // 2) or rank_two_i % 2:
            raise SchemaError("SwiGLU output must halve the rank-local feature axis", path=f"{path}.rank_output_shape")
        _validate_rank_shape(self.logical_input_shape, self.rank_input_shape, f"{path}.rank_input_shape")
        _validate_rank_shape(self.logical_output_shape, self.rank_output_shape, f"{path}.rank_output_shape")
        if self.dtype is not DType.FP16:
            raise SchemaError("current SwiGLU requires FP16", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class ResidualWorkload:
    logical_shape: tuple[int, int]
    rank_shape: tuple[int, int]
    dtype: DType

    def validate(self, path: str) -> None:
        _validate_shape(self.logical_shape, f"{path}.logical_shape", expected_rank=2)
        _validate_shape(self.rank_shape, f"{path}.rank_shape", expected_rank=2)
        _validate_rank_shape(self.logical_shape, self.rank_shape, f"{path}.rank_shape")
        if self.dtype is not DType.FP16:
            raise SchemaError("current residual add requires FP16", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class P2PByteWorkload:
    bytes: int
    dtype: DType

    def validate(self, path: str) -> None:
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        _validate_legacy_dtype(self.dtype, f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class AttentionWorkload:
    profile: ProfileKey
    mode: AttentionMode
    causal: bool
    query_tokens: int
    context_sum: int
    context_max: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rank_num_heads: int
    rank_num_kv_heads: int
    query_key_pairs: int
    logical_kv_read_bytes: int
    logical_kv_write_bytes: int
    rank_kv_read_bytes: int
    rank_kv_write_bytes: int
    dtype: DType
    exact_profile: Stage3StaticProfile | None = None

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        for name in (
            "query_tokens",
            "context_sum",
            "context_max",
            "hidden_size",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "rank_num_heads",
            "rank_num_kv_heads",
            "query_key_pairs",
        ):
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{name}")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise SchemaError("must equal num_heads * head_dim", path=f"{path}.hidden_size")
        if type(self.mode) is not AttentionMode:
            raise SchemaError("must be an AttentionMode", path=f"{path}.mode")
        if type(self.causal) is not bool or not self.causal:
            raise SchemaError("current attention requires causal=true", path=f"{path}.causal")
        if self.context_sum != self.profile.context_sum:
            raise SchemaError("must equal profile.context_sum", path=f"{path}.context_sum")
        if self.context_max != self.profile.context_max:
            raise SchemaError("must equal profile.context_max", path=f"{path}.context_max")
        for field_name in (
            "logical_kv_read_bytes",
            "logical_kv_write_bytes",
            "rank_kv_read_bytes",
            "rank_kv_write_bytes",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.num_kv_heads > self.num_heads or self.num_heads % self.num_kv_heads != 0:
            raise SchemaError("num_kv_heads must divide and not exceed num_heads", path=f"{path}.num_kv_heads")
        _validate_partitioned_extent(self.num_heads, self.rank_num_heads, f"{path}.rank_num_heads")
        _validate_partitioned_extent(self.num_kv_heads, self.rank_num_kv_heads, f"{path}.rank_num_kv_heads")
        if self.rank_num_heads % self.rank_num_kv_heads != 0:
            raise SchemaError("rank_num_kv_heads must divide rank_num_heads", path=f"{path}.rank_num_kv_heads")
        if self.num_heads // self.rank_num_heads != self.num_kv_heads // self.rank_num_kv_heads:
            raise SchemaError("query-head and KV-head rank shards must use the same TP degree", path=f"{path}.rank_num_kv_heads")
        tp = self.num_heads // self.rank_num_heads
        if self.mode is AttentionMode.TRAIN_FORWARD:
            if self.exact_profile is not None:
                raise SchemaError(
                    "train forward attention cannot carry an inference exact_profile",
                    path=f"{path}.exact_profile",
                )
            if self.profile.decode_tokens or not self.profile.prefill_tokens:
                raise SchemaError(
                    "train forward mode requires a pure forward profile",
                    path=f"{path}.mode",
                )
            if (
                self.query_tokens != self.profile.prefill_tokens
                or self.context_sum != self.query_tokens
                or self.profile.num_seqs * self.context_max
                != self.query_tokens
            ):
                raise SchemaError(
                    "train forward rows must equal num_seqs * fixed sequence length",
                    path=f"{path}.query_tokens",
                )
            expected_pairs = (
                self.profile.num_seqs
                * self.context_max
                * (self.context_max + 1)
                // 2
            )
            expected_read = 0
            expected_write = 0
        elif self.exact_profile is not None:
            if type(self.exact_profile) is not Stage3StaticProfile:
                raise SchemaError(
                    "must be a Stage3StaticProfile or null",
                    path=f"{path}.exact_profile",
                )
            self.exact_profile.validate(f"{path}.exact_profile")
            if self.exact_profile.key != self.profile:
                raise SchemaError(
                    "key must equal profile",
                    path=f"{path}.exact_profile.key",
                )
            expected_mode = {
                Stage3ProfileMode.PREFILL: AttentionMode.PREFILL,
                Stage3ProfileMode.DECODE: AttentionMode.DECODE,
                Stage3ProfileMode.MIXED: AttentionMode.MIXED,
            }[self.exact_profile.mode]
            if self.mode is not expected_mode:
                raise SchemaError(
                    f"must equal {expected_mode.value!r}", path=f"{path}.mode"
                )
            expected_tokens = (
                self.profile.prefill_tokens + self.profile.decode_tokens
            )
            if self.query_tokens != expected_tokens:
                raise SchemaError(
                    f"must equal {expected_tokens}",
                    path=f"{path}.query_tokens",
                )
            expected_pairs = self.exact_profile.capacity.query_key_pairs
            expected_read = (
                4
                * self.exact_profile.capacity.kv_read_tokens
                * self.num_kv_heads
                * self.head_dim
            )
            expected_write = (
                4
                * self.exact_profile.capacity.kv_write_tokens
                * self.num_kv_heads
                * self.head_dim
            )
        elif self.mode is AttentionMode.PREFILL:
            if self.profile.decode_tokens or not self.profile.prefill_tokens:
                raise SchemaError("prefill mode requires a pure prefill profile", path=f"{path}.mode")
            if self.query_tokens != self.profile.prefill_tokens:
                raise SchemaError("must equal profile.prefill_tokens", path=f"{path}.query_tokens")
            expected_pairs = self.query_tokens * (self.query_tokens + 1) // 2
            expected_read = 0
            expected_write = 4 * self.query_tokens * self.num_kv_heads * self.head_dim
        else:
            if self.mode is not AttentionMode.DECODE:
                raise SchemaError(
                    "mixed mode requires an exact_profile", path=f"{path}.mode"
                )
            if self.profile.prefill_tokens or not self.profile.decode_tokens:
                raise SchemaError("decode mode requires a pure decode profile", path=f"{path}.mode")
            if self.query_tokens != self.profile.decode_tokens:
                raise SchemaError("must equal profile.decode_tokens", path=f"{path}.query_tokens")
            expected_pairs = self.context_sum
            expected_read = 4 * self.context_sum * self.num_kv_heads * self.head_dim
            expected_write = 4 * self.query_tokens * self.num_kv_heads * self.head_dim
        if self.query_key_pairs != expected_pairs:
            raise SchemaError(f"must equal {expected_pairs}", path=f"{path}.query_key_pairs")
        if self.logical_kv_read_bytes != expected_read:
            raise SchemaError(f"must equal {expected_read}", path=f"{path}.logical_kv_read_bytes")
        if self.logical_kv_write_bytes != expected_write:
            raise SchemaError(f"must equal {expected_write}", path=f"{path}.logical_kv_write_bytes")
        if expected_read % tp or expected_write % tp:
            raise SchemaError("logical KV bytes must divide evenly across TP", path=path)
        if self.rank_kv_read_bytes != expected_read // tp:
            raise SchemaError(f"must equal {expected_read // tp}", path=f"{path}.rank_kv_read_bytes")
        if self.rank_kv_write_bytes != expected_write // tp:
            raise SchemaError(f"must equal {expected_write // tp}", path=f"{path}.rank_kv_write_bytes")
        if self.dtype is not DType.FP16:
            raise SchemaError("current attention requires FP16", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class EmbeddingWorkload:
    profile: ProfileKey
    logical_index_shape: tuple[int, ...]
    rank_index_shape: tuple[int, ...]
    logical_table_shape: tuple[int, ...]
    rank_table_shape: tuple[int, ...]
    logical_output_shape: tuple[int, ...]
    rank_output_shape: tuple[int, ...]
    table_placement: EmbeddingTablePlacement
    index_dtype: DType
    table_dtype: DType
    output_dtype: DType

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        for field_name, expected_rank in (
            ("logical_index_shape", 1),
            ("rank_index_shape", 1),
            ("logical_table_shape", 2),
            ("rank_table_shape", 2),
            ("logical_output_shape", 2),
            ("rank_output_shape", 2),
        ):
            _validate_shape(
                getattr(self, field_name),
                f"{path}.{field_name}",
                expected_rank=expected_rank,
            )
        tokens = self.profile.prefill_tokens + self.profile.decode_tokens
        if self.logical_index_shape != (tokens,):
            raise SchemaError(
                f"must equal ({tokens},)", path=f"{path}.logical_index_shape"
            )
        rank_tokens = self.rank_index_shape[0]
        _validate_partitioned_extent(tokens, rank_tokens, f"{path}.rank_index_shape[0]")
        vocabulary, hidden_size = self.logical_table_shape
        if self.table_placement is not EmbeddingTablePlacement.REPLICATED:
            raise SchemaError(
                "current embedding requires a replicated table",
                path=f"{path}.table_placement",
            )
        if self.rank_table_shape != self.logical_table_shape:
            raise SchemaError(
                "replicated embedding requires rank_table_shape == logical_table_shape",
                path=f"{path}.rank_table_shape",
            )
        if self.logical_output_shape != (tokens, hidden_size):
            raise SchemaError(
                f"must equal {(tokens, hidden_size)!r}",
                path=f"{path}.logical_output_shape",
            )
        if self.rank_output_shape != (rank_tokens, hidden_size):
            raise SchemaError(
                f"must equal {(rank_tokens, hidden_size)!r}",
                path=f"{path}.rank_output_shape",
            )
        if vocabulary == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.logical_table_shape[0]")
        if self.index_dtype is not DType.INT32:
            raise SchemaError("embedding indices require INT32", path=f"{path}.index_dtype")
        if self.table_dtype is not DType.FP16:
            raise SchemaError("current embedding table requires FP16", path=f"{path}.table_dtype")
        if self.output_dtype is not DType.FP16:
            raise SchemaError("current embedding output requires FP16", path=f"{path}.output_dtype")


@dataclass(frozen=True, slots=True)
class RopeQkWorkload:
    """Packed Q/K/V input with Q and K rotated and V passed through unchanged."""

    profile: ProfileKey
    logical_input_shape: tuple[int, ...]
    rank_input_shape: tuple[int, ...]
    logical_output_shape: tuple[int, ...]
    rank_output_shape: tuple[int, ...]
    packed_layout: PackedQkvLayout
    num_heads: int
    num_kv_heads: int
    rank_num_heads: int
    rank_num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    max_position_embeddings: int
    dtype: DType

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        for field_name in (
            "logical_input_shape",
            "rank_input_shape",
            "logical_output_shape",
            "rank_output_shape",
        ):
            _validate_shape(getattr(self, field_name), f"{path}.{field_name}", expected_rank=2)
        if self.logical_output_shape != self.logical_input_shape:
            raise SchemaError("ROPE must preserve logical packed shape", path=f"{path}.logical_output_shape")
        if self.rank_output_shape != self.rank_input_shape:
            raise SchemaError("ROPE must preserve rank-local packed shape", path=f"{path}.rank_output_shape")
        if self.packed_layout is not PackedQkvLayout.Q_K_V:
            raise SchemaError("current ROPE requires packed Q_K_V layout", path=f"{path}.packed_layout")
        for field_name in (
            "num_heads",
            "num_kv_heads",
            "rank_num_heads",
            "rank_num_kv_heads",
            "head_dim",
            "rotary_dim",
            "max_position_embeddings",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{field_name}")
        if self.num_kv_heads > self.num_heads or self.num_heads % self.num_kv_heads:
            raise SchemaError("num_kv_heads must divide and not exceed num_heads", path=f"{path}.num_kv_heads")
        _validate_partitioned_extent(self.num_heads, self.rank_num_heads, f"{path}.rank_num_heads")
        _validate_partitioned_extent(self.num_kv_heads, self.rank_num_kv_heads, f"{path}.rank_num_kv_heads")
        if self.rank_num_heads % self.rank_num_kv_heads:
            raise SchemaError("rank_num_kv_heads must divide rank_num_heads", path=f"{path}.rank_num_kv_heads")
        if self.num_heads // self.rank_num_heads != self.num_kv_heads // self.rank_num_kv_heads:
            raise SchemaError("query-head and KV-head shards must use the same TP degree", path=f"{path}.rank_num_kv_heads")
        if self.rotary_dim != self.head_dim or self.rotary_dim % 2:
            raise SchemaError("current ROPE requires even rotary_dim == head_dim", path=f"{path}.rotary_dim")
        if type(self.rope_theta) is not float or not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.rope_theta")
        if self.profile.context_max > self.max_position_embeddings:
            raise SchemaError("must cover profile.context_max", path=f"{path}.max_position_embeddings")
        tokens = self.profile.prefill_tokens + self.profile.decode_tokens
        logical_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        rank_width = (self.rank_num_heads + 2 * self.rank_num_kv_heads) * self.head_dim
        if self.logical_input_shape != (tokens, logical_width):
            raise SchemaError(
                f"must equal {(tokens, logical_width)!r}", path=f"{path}.logical_input_shape"
            )
        if self.rank_input_shape != (tokens, rank_width):
            raise SchemaError(
                f"must equal {(tokens, rank_width)!r}", path=f"{path}.rank_input_shape"
            )
        if self.dtype is not DType.FP16:
            raise SchemaError("current ROPE requires FP16", path=f"{path}.dtype")


@dataclass(frozen=True, slots=True)
class GreedySampleWorkload:
    profile: ProfileKey
    mode: SamplingMode
    row_selection: SampleRowSelection
    tp_degree: int
    logical_logits_shape: tuple[int, ...]
    rank_logits_shape: tuple[int, ...]
    logical_output_shape: tuple[int, ...]
    rank_output_shape: tuple[int, ...]
    sample_count: int
    comparisons: int
    logits_dtype: DType
    output_dtype: DType

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        if self.mode is not SamplingMode.GREEDY:
            raise SchemaError("current sampling requires GREEDY mode", path=f"{path}.mode")
        if self.row_selection is not SampleRowSelection.LAST_PER_SEQUENCE:
            raise SchemaError("current sampling requires LAST_PER_SEQUENCE rows", path=f"{path}.row_selection")
        validate_uint64(self.tp_degree, f"{path}.tp_degree")
        if self.tp_degree != 1:
            raise UnsupportedFeatureError("TP-sharded greedy sampling is not implemented", path=f"{path}.tp_degree")
        for field_name, expected_rank in (
            ("logical_logits_shape", 2),
            ("rank_logits_shape", 2),
            ("logical_output_shape", 1),
            ("rank_output_shape", 1),
        ):
            _validate_shape(getattr(self, field_name), f"{path}.{field_name}", expected_rank=expected_rank)
        tokens = self.profile.prefill_tokens + self.profile.decode_tokens
        if self.logical_logits_shape[0] != tokens:
            raise SchemaError(f"row count must equal {tokens}", path=f"{path}.logical_logits_shape[0]")
        vocabulary = self.logical_logits_shape[1]
        if vocabulary <= 1:
            raise SchemaError("vocabulary must be greater than one", path=f"{path}.logical_logits_shape[1]")
        if self.rank_logits_shape != self.logical_logits_shape:
            raise SchemaError("TP1 requires rank_logits_shape == logical_logits_shape", path=f"{path}.rank_logits_shape")
        expected_count = self.profile.num_seqs
        if self.sample_count != expected_count:
            raise SchemaError(f"must equal {expected_count}", path=f"{path}.sample_count")
        if self.logical_output_shape != (expected_count,) or self.rank_output_shape != (expected_count,):
            raise SchemaError("greedy output must contain one sample per sequence", path=f"{path}.logical_output_shape")
        expected_comparisons = expected_count * (vocabulary - 1)
        if self.comparisons != expected_comparisons:
            raise SchemaError(f"must equal {expected_comparisons}", path=f"{path}.comparisons")
        if self.logits_dtype is not DType.FP16:
            raise SchemaError("current greedy logits require FP16", path=f"{path}.logits_dtype")
        if self.output_dtype is not DType.INT32:
            raise SchemaError("greedy sample ids require INT32", path=f"{path}.output_dtype")


@dataclass(frozen=True, slots=True)
class CrossEntropyForwardWorkload:
    profile: ProfileKey
    reduction: CrossEntropyReduction
    logical_logits_shape: tuple[int, ...]
    rank_logits_shape: tuple[int, ...]
    logical_label_shape: tuple[int, ...]
    rank_label_shape: tuple[int, ...]
    logical_loss_shape: tuple[int, ...]
    rank_loss_shape: tuple[int, ...]
    logits_dtype: DType
    label_dtype: DType
    loss_dtype: DType

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        if self.reduction in (CrossEntropyReduction.SUM, CrossEntropyReduction.MEAN):
            raise UnsupportedFeatureError("reduced cross entropy is not implemented", path=f"{path}.reduction")
        if self.reduction is not CrossEntropyReduction.NONE:
            raise SchemaError("must be a CrossEntropyReduction", path=f"{path}.reduction")
        for field_name, expected_rank in (
            ("logical_logits_shape", 2),
            ("rank_logits_shape", 2),
            ("logical_label_shape", 1),
            ("rank_label_shape", 1),
            ("logical_loss_shape", 1),
            ("rank_loss_shape", 1),
        ):
            _validate_shape(getattr(self, field_name), f"{path}.{field_name}", expected_rank=expected_rank)
        tokens = self.profile.prefill_tokens + self.profile.decode_tokens
        if self.logical_logits_shape[0] != tokens:
            raise SchemaError(f"row count must equal {tokens}", path=f"{path}.logical_logits_shape[0]")
        vocabulary = self.logical_logits_shape[1]
        if vocabulary <= 1:
            raise SchemaError("vocabulary must be greater than one", path=f"{path}.logical_logits_shape[1]")
        rank_tokens, rank_vocabulary = self.rank_logits_shape
        _validate_partitioned_extent(tokens, rank_tokens, f"{path}.rank_logits_shape[0]")
        if rank_vocabulary != vocabulary:
            raise SchemaError("vocabulary axis must be replicated", path=f"{path}.rank_logits_shape[1]")
        if self.logical_label_shape != (tokens,) or self.logical_loss_shape != (tokens,):
            raise SchemaError("labels and unreduced loss must have one value per logical row", path=f"{path}.logical_label_shape")
        if self.rank_label_shape != (rank_tokens,) or self.rank_loss_shape != (rank_tokens,):
            raise SchemaError("labels and unreduced loss must match rank-local rows", path=f"{path}.rank_label_shape")
        if self.logits_dtype is not DType.FP16:
            raise SchemaError("current CE logits require FP16", path=f"{path}.logits_dtype")
        if self.label_dtype is not DType.INT32:
            raise SchemaError("CE labels require INT32", path=f"{path}.label_dtype")
        if self.loss_dtype is not DType.FP32:
            raise SchemaError("current CE loss requires FP32", path=f"{path}.loss_dtype")


@dataclass(frozen=True, slots=True)
class CrossEntropyBackwardWorkload:
    profile: ProfileKey
    reduction: CrossEntropyReduction
    logical_logits_shape: tuple[int, ...]
    rank_logits_shape: tuple[int, ...]
    logical_label_shape: tuple[int, ...]
    rank_label_shape: tuple[int, ...]
    logical_loss_gradient_shape: tuple[int, ...]
    rank_loss_gradient_shape: tuple[int, ...]
    logical_logits_gradient_shape: tuple[int, ...]
    rank_logits_gradient_shape: tuple[int, ...]
    logits_dtype: DType
    label_dtype: DType
    loss_gradient_dtype: DType
    logits_gradient_dtype: DType

    def validate(self, path: str) -> None:
        self.profile.validate(f"{path}.profile")
        if self.reduction in (
            CrossEntropyReduction.SUM,
            CrossEntropyReduction.MEAN,
        ):
            raise UnsupportedFeatureError(
                "reduced cross entropy backward is not implemented",
                path=f"{path}.reduction",
            )
        if self.reduction is not CrossEntropyReduction.NONE:
            raise SchemaError(
                "must be a CrossEntropyReduction", path=f"{path}.reduction"
            )
        for field_name, expected_rank in (
            ("logical_logits_shape", 2),
            ("rank_logits_shape", 2),
            ("logical_label_shape", 1),
            ("rank_label_shape", 1),
            ("logical_loss_gradient_shape", 1),
            ("rank_loss_gradient_shape", 1),
            ("logical_logits_gradient_shape", 2),
            ("rank_logits_gradient_shape", 2),
        ):
            _validate_shape(
                getattr(self, field_name),
                f"{path}.{field_name}",
                expected_rank=expected_rank,
            )
        tokens = self.profile.prefill_tokens + self.profile.decode_tokens
        if self.logical_logits_shape[0] != tokens:
            raise SchemaError(
                f"row count must equal {tokens}",
                path=f"{path}.logical_logits_shape[0]",
            )
        vocabulary = self.logical_logits_shape[1]
        if vocabulary <= 1:
            raise SchemaError(
                "vocabulary must be greater than one",
                path=f"{path}.logical_logits_shape[1]",
            )
        rank_tokens, rank_vocabulary = self.rank_logits_shape
        _validate_partitioned_extent(
            tokens, rank_tokens, f"{path}.rank_logits_shape[0]"
        )
        if rank_vocabulary != vocabulary:
            raise SchemaError(
                "vocabulary axis must be replicated",
                path=f"{path}.rank_logits_shape[1]",
            )
        if (
            self.logical_label_shape != (tokens,)
            or self.logical_loss_gradient_shape != (tokens,)
            or self.logical_logits_gradient_shape != self.logical_logits_shape
        ):
            raise SchemaError(
                "logical CE backward tensors are not exact", path=path
            )
        if (
            self.rank_label_shape != (rank_tokens,)
            or self.rank_loss_gradient_shape != (rank_tokens,)
            or self.rank_logits_gradient_shape != self.rank_logits_shape
        ):
            raise SchemaError(
                "rank-local CE backward tensors are not exact", path=path
            )
        for field_name, expected in (
            ("logits_dtype", DType.FP16),
            ("label_dtype", DType.INT32),
            ("loss_gradient_dtype", DType.FP32),
            ("logits_gradient_dtype", DType.FP16),
        ):
            if getattr(self, field_name) is not expected:
                raise SchemaError(
                    f"must be {expected.value}", path=f"{path}.{field_name}"
                )


@dataclass(frozen=True, slots=True)
class SgdUpdateWorkload:
    logical_weight_shape: tuple[int, ...]
    rank_weight_shape: tuple[int, ...]
    logical_gradient_shape: tuple[int, ...]
    rank_gradient_shape: tuple[int, ...]
    logical_updated_weight_shape: tuple[int, ...]
    rank_updated_weight_shape: tuple[int, ...]
    element_count: int
    learning_rate: float
    momentum: float
    weight_dtype: DType
    gradient_dtype: DType
    updated_weight_dtype: DType

    def validate(self, path: str) -> None:
        for field_name in (
            "logical_weight_shape",
            "rank_weight_shape",
            "logical_gradient_shape",
            "rank_gradient_shape",
            "logical_updated_weight_shape",
            "rank_updated_weight_shape",
        ):
            _validate_shape(
                getattr(self, field_name), f"{path}.{field_name}", expected_rank=2
            )
        if not (
            self.logical_gradient_shape
            == self.logical_updated_weight_shape
            == self.logical_weight_shape
            and self.rank_gradient_shape
            == self.rank_updated_weight_shape
            == self.rank_weight_shape
            == self.logical_weight_shape
        ):
            raise SchemaError(
                "S2-Lite SGD requires replicated equal weight/gradient shapes",
                path=path,
            )
        validate_uint64(self.element_count, f"{path}.element_count")
        if self.element_count != math.prod(self.logical_weight_shape):
            raise SchemaError(
                "must equal the logical weight element count",
                path=f"{path}.element_count",
            )
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float", path=f"{path}.learning_rate"
            )
        if type(self.momentum) is not float or self.momentum != 0.0:
            raise SchemaError(
                "S2-Lite SGD requires momentum=0", path=f"{path}.momentum"
            )
        for field_name, expected in (
            ("weight_dtype", DType.FP16),
            ("gradient_dtype", DType.FP32),
            ("updated_weight_dtype", DType.FP16),
        ):
            if getattr(self, field_name) is not expected:
                raise SchemaError(
                    f"must be {expected.value}", path=f"{path}.{field_name}"
                )


@dataclass(frozen=True, slots=True)
class CollectiveWorkload:
    collective: CollectiveKind
    reduce_op: ReduceOp | None
    mesh_axes: tuple[MeshAxisName, ...]
    participant_count: int
    reduction_mesh_axes: tuple[MeshAxisName, ...]
    scatter_tensor_axis: int | None
    gather_tensor_axis: int | None
    logical_tensor_bytes: int
    rank_input_bytes: int
    rank_output_bytes: int
    rank_logical_payload_bytes: int
    group_logical_payload_bytes: int
    dtype: DType
    role: CollectiveRole
    input_layout: str
    output_layout: str

    def validate(self, path: str) -> None:
        validate_uint64(self.participant_count, f"{path}.participant_count")
        if self.participant_count <= 1:
            raise SchemaError("must be greater than one", path=f"{path}.participant_count")
        for field_name in (
            "logical_tensor_bytes",
            "rank_input_bytes",
            "rank_output_bytes",
            "rank_logical_payload_bytes",
            "group_logical_payload_bytes",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{field_name}")
        if not self.mesh_axes or len(set(self.mesh_axes)) != len(self.mesh_axes):
            raise SchemaError("must contain unique mesh axes", path=f"{path}.mesh_axes")
        if len(set(self.reduction_mesh_axes)) != len(self.reduction_mesh_axes):
            raise SchemaError("contains duplicate mesh axes", path=f"{path}.reduction_mesh_axes")
        if not set(self.reduction_mesh_axes).issubset(self.mesh_axes):
            raise SchemaError("must be a subset of mesh_axes", path=f"{path}.reduction_mesh_axes")
        for field_name in ("scatter_tensor_axis", "gather_tensor_axis"):
            axis = getattr(self, field_name)
            if axis is not None:
                validate_uint64(axis, f"{path}.{field_name}")
        _validate_legacy_dtype(self.dtype, f"{path}.dtype")
        validate_nonempty(self.input_layout, f"{path}.input_layout")
        validate_nonempty(self.output_layout, f"{path}.output_layout")

        if self.collective is CollectiveKind.ALL_GATHER:
            self._validate_all_gather(path)
        elif self.collective is CollectiveKind.REDUCE_SCATTER:
            self._validate_reduce_scatter(path)
        elif self.collective in (CollectiveKind.ALL_REDUCE, CollectiveKind.ALL_TO_ALL):
            raise UnsupportedFeatureError(
                "N2a workload schema supports only AllGather and ReduceScatter",
                path=f"{path}.collective",
            )
        else:
            raise SchemaError("unsupported collective kind", path=f"{path}.collective")

    def _require_no_reduce(self, path: str) -> None:
        if self.reduce_op is not None:
            raise SchemaError("must be null for a non-reduction collective", path=f"{path}.reduce_op")
        if self.reduction_mesh_axes:
            raise SchemaError("must be empty for a non-reduction collective", path=f"{path}.reduction_mesh_axes")

    def _require_reduce(self, path: str) -> None:
        if self.reduce_op is None:
            raise SchemaError("is required for a reduction collective", path=f"{path}.reduce_op")
        if self.reduction_mesh_axes != self.mesh_axes:
            raise SchemaError("must exactly equal mesh_axes for this reduction collective", path=f"{path}.reduction_mesh_axes")

    def _validate_all_gather(self, path: str) -> None:
        self._require_no_reduce(path)
        if self.scatter_tensor_axis is not None or self.gather_tensor_axis is None:
            raise SchemaError("AllGather requires only gather_tensor_axis", path=path)
        if self.rank_output_bytes != self.logical_tensor_bytes:
            raise SchemaError("AllGather rank output must equal the logical tensor", path=f"{path}.rank_output_bytes")
        if self.logical_tensor_bytes % self.participant_count != 0:
            raise SchemaError("logical tensor bytes must divide evenly across participants", path=f"{path}.logical_tensor_bytes")
        expected_rank_input = self.logical_tensor_bytes // self.participant_count
        expected = self.rank_output_bytes - self.rank_input_bytes
        if self.rank_input_bytes != expected_rank_input:
            raise SchemaError("AllGather rank input must be one participant shard", path=f"{path}.rank_input_bytes")
        if self.rank_logical_payload_bytes != expected or self.group_logical_payload_bytes != expected * self.participant_count:
            raise SchemaError("AllGather logical payloads are inconsistent", path=f"{path}.rank_logical_payload_bytes")

    def _validate_reduce_scatter(self, path: str) -> None:
        self._require_reduce(path)
        if self.reduce_op is not ReduceOp.SUM:
            raise UnsupportedFeatureError("N2a ReduceScatter supports SUM only", path=f"{path}.reduce_op")
        if self.scatter_tensor_axis is None or self.gather_tensor_axis is not None:
            raise SchemaError("ReduceScatter requires only scatter_tensor_axis", path=path)
        if self.rank_input_bytes != self.logical_tensor_bytes:
            raise SchemaError("ReduceScatter rank input must equal the logical tensor", path=f"{path}.rank_input_bytes")
        if self.logical_tensor_bytes % self.participant_count != 0:
            raise SchemaError("logical tensor bytes must divide evenly across participants", path=f"{path}.logical_tensor_bytes")
        expected_rank_output = self.logical_tensor_bytes // self.participant_count
        expected = self.rank_input_bytes - self.rank_output_bytes
        if self.rank_output_bytes != expected_rank_output:
            raise SchemaError("ReduceScatter rank output must be one participant shard", path=f"{path}.rank_output_bytes")
        if self.rank_logical_payload_bytes != expected or self.group_logical_payload_bytes != expected * self.participant_count:
            raise SchemaError("ReduceScatter logical payloads are inconsistent", path=f"{path}.rank_logical_payload_bytes")


NodeWorkload = (
    GemmWorkload
    | NormWorkload
    | ElementwiseWorkload
    | P2PByteWorkload
    | RmsNormWorkload
    | SwiGluWorkload
    | ResidualWorkload
    | AttentionWorkload
    | EmbeddingWorkload
    | RopeQkWorkload
    | GreedySampleWorkload
    | CrossEntropyForwardWorkload
    | CrossEntropyBackwardWorkload
    | SgdUpdateWorkload
    | CollectiveWorkload
)


@dataclass(frozen=True, slots=True)
class NodeMath:
    accumulation_dtype: DType
    numerical_policy: NumericalPolicy

    def validate(self, path: str) -> None:
        # Workload-specific contract/scatter/gather axes deliberately do not
        # live in this common numerical policy.
        _validate_legacy_dtype(self.accumulation_dtype, f"{path}.accumulation_dtype")
        if type(self.numerical_policy) is not NumericalPolicy:
            raise SchemaError("must be a NumericalPolicy", path=f"{path}.numerical_policy")


@dataclass(frozen=True, slots=True)
class NodeEffects:
    kind: EffectKind
    effect_token: str | None
    alias_set: str | None

    def validate(self, path: str) -> None:
        if self.kind is EffectKind.PURE and self.effect_token is not None:
            raise SchemaError("pure nodes cannot carry an effect token", path=f"{path}.effect_token")
        if self.kind is not EffectKind.PURE and self.effect_token is None:
            raise SchemaError("non-pure nodes require an effect token", path=f"{path}.effect_token")
        if self.effect_token is not None:
            validate_nonempty(self.effect_token, f"{path}.effect_token")
        if self.alias_set is not None:
            validate_nonempty(self.alias_set, f"{path}.alias_set")


@dataclass(frozen=True, slots=True)
class LogicalNode:
    id: str
    instance_id: str
    kind: OpKind
    phase: OpPhase
    stage: int
    mesh_ref: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    workload: NodeWorkload
    math: NodeMath
    effects: NodeEffects
    impl_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.instance_id, f"{path}.instance_id")
        if type(self.kind) is not OpKind:
            raise SchemaError("must be an OpKind", path=f"{path}.kind")
        if type(self.phase) is not OpPhase:
            raise SchemaError("must be an OpPhase", path=f"{path}.phase")
        validate_uint64(self.stage, f"{path}.stage")
        validate_nonempty(self.mesh_ref, f"{path}.mesh_ref")
        validate_nonempty(self.impl_ref, f"{path}.impl_ref")
        for field_name in ("inputs", "outputs"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")
        expected_types = {
            OpKind.GEMM: GemmWorkload,
            OpKind.ELEMENTWISE: (SwiGluWorkload, ResidualWorkload),
            OpKind.NORM: RmsNormWorkload,
            OpKind.ATTENTION: AttentionWorkload,
            OpKind.COLLECTIVE: CollectiveWorkload,
            OpKind.P2P: P2PByteWorkload,
            OpKind.EMBEDDING: EmbeddingWorkload,
            OpKind.ROPE: RopeQkWorkload,
            OpKind.SAMPLING: GreedySampleWorkload,
            OpKind.CE_FORWARD: CrossEntropyForwardWorkload,
            OpKind.CE_BACKWARD: CrossEntropyBackwardWorkload,
            OpKind.OPTIMIZER_UPDATE: SgdUpdateWorkload,
        }[self.kind]
        expected_types = expected_types if type(expected_types) is tuple else (expected_types,)
        if type(self.workload) not in expected_types:
            raise SchemaError(
                f"{self.kind.value!r} requires one of {[item.__name__ for item in expected_types]!r}", path=f"{path}.workload"
            )
        self.workload.validate(f"{path}.workload")
        self.math.validate(f"{path}.math")
        self.effects.validate(f"{path}.effects")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: str
    kind: EdgeKind
    source_node: str
    destination_node: str
    value_id: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.source_node, f"{path}.source_node")
        validate_nonempty(self.destination_node, f"{path}.destination_node")
        if self.source_node == self.destination_node:
            raise SchemaError("self edges are not allowed", path=path)
        if self.kind is EdgeKind.DATA and self.value_id is None:
            raise SchemaError("data edges require value_id", path=f"{path}.value_id")
        if self.kind is EdgeKind.CONTROL and self.value_id is not None:
            raise SchemaError("control edges require value_id=null", path=f"{path}.value_id")
        if self.value_id is not None:
            validate_nonempty(self.value_id, f"{path}.value_id")


@dataclass(frozen=True, slots=True)
class FusionSemanticContract:
    tile_domain: tuple[str, ...]
    reduction_axes: tuple[int, ...]
    input_layouts: tuple[str, ...]
    output_layout: str
    numerical_policy: NumericalPolicy

    def validate(self, path: str) -> None:
        for index, axis in enumerate(self.tile_domain):
            validate_nonempty(axis, f"{path}.tile_domain[{index}]")
        for index, axis in enumerate(self.reduction_axes):
            validate_uint64(axis, f"{path}.reduction_axes[{index}]")
        for index, layout in enumerate(self.input_layouts):
            validate_nonempty(layout, f"{path}.input_layouts[{index}]")
        validate_nonempty(self.output_layout, f"{path}.output_layout")


@dataclass(frozen=True, slots=True)
class FusionCandidate:
    id: str
    members: tuple[str, ...]
    boundary_inputs: tuple[str, ...]
    boundary_outputs: tuple[str, ...]
    semantic_contract: FusionSemanticContract
    impl: FusionImpl
    origin: FusionOrigin

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        if not self.members or len(set(self.members)) != len(self.members):
            raise SchemaError("must contain unique member ids", path=f"{path}.members")
        for field_name in ("members", "boundary_inputs", "boundary_outputs"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")
        self.semantic_contract.validate(f"{path}.semantic_contract")
        if self.impl is not FusionImpl.NONE:
            raise SchemaError(
                "IR-0 candidates cannot select an implementation before FusionPartition",
                path=f"{path}.impl",
            )


@dataclass(frozen=True, slots=True)
class TrainStructure:
    micro_batch_count: int
    pp_schedule: PipelineSchedule
    interleave_chunks: int
    recompute: RecomputeMode

    def validate(self, path: str) -> None:
        for name in ("micro_batch_count", "interleave_chunks"):
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{name}")


def validate_value_graph(
    nodes: tuple[LogicalNode, ...],
    values: tuple[TensorValue, ...],
    edges: tuple[GraphEdge, ...],
    *,
    path: str,
) -> tuple[dict[str, object], dict[str, object]]:
    node_index = validate_unique_ids(nodes, f"{path}.nodes")
    value_index = validate_unique_ids(values, f"{path}.values")
    validate_unique_ids(edges, f"{path}.edges")
    for index, node in enumerate(nodes):
        node.validate(f"{path}.nodes[{index}]")
        for field_name in ("inputs", "outputs"):
            for ref in getattr(node, field_name):
                if ref not in value_index:
                    raise SchemaError(
                        f"dangling value reference {ref!r}", path=f"{path}.nodes[{index}].{field_name}"
                    )
    for index, value in enumerate(values):
        value.validate(f"{path}.values[{index}]")
        if value.producer is not None:
            producer = node_index.get(value.producer)
            if producer is None:
                raise SchemaError(
                    f"dangling producer {value.producer!r}", path=f"{path}.values[{index}].producer"
                )
            if value.id not in producer.outputs:
                raise SchemaError(
                    "producer node does not list this value as an output",
                    path=f"{path}.values[{index}].producer",
                )
        for consumer_id in value.consumers:
            consumer = node_index.get(consumer_id)
            if consumer is None:
                raise SchemaError(
                    f"dangling consumer {consumer_id!r}", path=f"{path}.values[{index}].consumers"
                )
            if value.id not in consumer.inputs:
                raise SchemaError(
                    "consumer node does not list this value as an input",
                    path=f"{path}.values[{index}].consumers",
                )
    for index, node in enumerate(nodes):
        for value_id in node.inputs:
            value = value_index[value_id]
            if node.id not in value.consumers:
                raise SchemaError(
                    "input value is missing this node from consumers",
                    path=f"{path}.nodes[{index}].inputs",
                )
        for value_id in node.outputs:
            value = value_index[value_id]
            if value.producer != node.id:
                raise SchemaError(
                    "output value names a different producer",
                    path=f"{path}.nodes[{index}].outputs",
                )
    expected_data_edges = {
        (value.producer, consumer, value.id)
        for value in values
        if value.producer is not None
        for consumer in value.consumers
    }
    actual_data_edges: set[tuple[str, str, str]] = set()
    for index, edge in enumerate(edges):
        edge.validate(f"{path}.edges[{index}]")
        if edge.source_node not in node_index or edge.destination_node not in node_index:
            raise SchemaError("edge contains a dangling node reference", path=f"{path}.edges[{index}]")
        if edge.kind is EdgeKind.DATA:
            assert edge.value_id is not None
            if edge.value_id not in value_index:
                raise SchemaError("edge contains a dangling value reference", path=f"{path}.edges[{index}].value_id")
            triple = (edge.source_node, edge.destination_node, edge.value_id)
            if triple in actual_data_edges:
                raise SchemaError("duplicate data edge", path=f"{path}.edges[{index}]")
            actual_data_edges.add(triple)
    if actual_data_edges != expected_data_edges:
        missing = sorted(expected_data_edges - actual_data_edges)
        extra = sorted(actual_data_edges - expected_data_edges)
        raise SchemaError(
            f"data edges do not match the value table; missing={missing!r}, extra={extra!r}",
            path=f"{path}.edges",
        )
    return node_index, value_index


@dataclass(frozen=True, slots=True)
class InstanceProfileBinding:
    instance_ref: str
    profile: ProfileKey

    def validate(self, path: str = "instance_profile") -> None:
        validate_nonempty(self.instance_ref, f"{path}.instance_ref")
        if type(self.profile) is not ProfileKey:
            raise SchemaError("must be a ProfileKey", path=f"{path}.profile")
        self.profile.validate(f"{path}.profile")


@dataclass(frozen=True, slots=True)
class NodeProfileBinding:
    node_ref: str
    profile: ProfileKey

    def validate(self, path: str = "node_profile") -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.profile) is not ProfileKey:
            raise SchemaError("must be a ProfileKey", path=f"{path}.profile")
        self.profile.validate(f"{path}.profile")


@dataclass(frozen=True, slots=True)
class IR0:
    schema_version: str
    producer_pass: str
    id: str
    job: JobKind
    instances: tuple[LogicalInstance, ...]
    nodes: tuple[LogicalNode, ...]
    values: tuple[TensorValue, ...]
    edges: tuple[GraphEdge, ...]
    fusion_candidates: tuple[FusionCandidate, ...]
    profile: ProfileKey
    train: TrainStructure | None
    instance_profiles: tuple[InstanceProfileBinding, ...] = ()
    node_profiles: tuple[NodeProfileBinding, ...] = ()
    pd_plan_id: str | None = None
    persistent_states: tuple[PersistentStateDecl, ...] = ()
    state_accesses: tuple[StateAccess, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        job: JobKind,
        instances: tuple[LogicalInstance, ...],
        nodes: tuple[LogicalNode, ...],
        values: tuple[TensorValue, ...],
        edges: tuple[GraphEdge, ...],
        fusion_candidates: tuple[FusionCandidate, ...],
        profile: ProfileKey,
        train: TrainStructure | None = None,
        instance_profiles: tuple[InstanceProfileBinding, ...] = (),
        node_profiles: tuple[NodeProfileBinding, ...] = (),
        pd_plan_id: str | None = None,
        persistent_states: tuple[PersistentStateDecl, ...] = (),
        state_accesses: tuple[StateAccess, ...] = (),
    ) -> "IR0":
        persistent_states = tuple(
            sorted(
                persistent_states,
                key=lambda item: (item.identity.id, item.id),
            )
        )
        state_accesses = tuple(
            sorted(
                state_accesses,
                key=lambda item: (
                    item.node_ref, item.state_ref, item.rank, item.id
                ),
            )
        )
        semantic_key = {
            "job": job,
            "instances": instances,
            "nodes": nodes,
            "values": values,
            "edges": edges,
            "fusion_candidates": fusion_candidates,
            "persistent_states": persistent_states,
            "state_accesses": state_accesses,
            "profile": profile,
            "train": train,
            "instance_profiles": instance_profiles,
            "node_profiles": node_profiles,
            "pd_plan_id": pd_plan_id,
        }
        return cls(
            schema_version=IR0_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("ir0", semantic_key, schema_version=IR0_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "job": self.job,
            "instances": self.instances,
            "nodes": self.nodes,
            "values": self.values,
            "edges": self.edges,
            "fusion_candidates": self.fusion_candidates,
            "persistent_states": self.persistent_states,
            "state_accesses": self.state_accesses,
            "profile": self.profile,
            "train": self.train,
            "instance_profiles": self.instance_profiles,
            "node_profiles": self.node_profiles,
            "pd_plan_id": self.pd_plan_id,
        }

    def validate(self, path: str = "ir0") -> None:
        if self.schema_version != IR0_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        expected_id = stable_artifact_id("ir0", self._semantic_key(), schema_version=IR0_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")
        instance_index = validate_unique_ids(self.instances, f"{path}.instances")
        profiles_by_instance: dict[str, set[ProfileKey]]
        if self.instance_profiles:
            if self.pd_plan_id is None:
                raise SchemaError(
                    "is required with instance_profiles",
                    path=f"{path}.pd_plan_id",
                )
            validate_nonempty(self.pd_plan_id, f"{path}.pd_plan_id")
            if tuple(
                sorted(
                    self.instance_profiles,
                    key=lambda item: (
                        item.instance_ref,
                        item.profile.stable_id(),
                    ),
                )
            ) != self.instance_profiles:
                raise SchemaError(
                    "must use canonical instance_ref/profile order",
                    path=f"{path}.instance_profiles",
                )
            profiles_by_instance = {}
            binding_keys: set[tuple[str, str]] = set()
            for index, binding in enumerate(self.instance_profiles):
                binding_path = f"{path}.instance_profiles[{index}]"
                if type(binding) is not InstanceProfileBinding:
                    raise SchemaError(
                        "must be an InstanceProfileBinding", path=binding_path
                    )
                binding.validate(binding_path)
                if binding.instance_ref not in instance_index:
                    raise SchemaError(
                        "references a dangling instance",
                        path=f"{binding_path}.instance_ref",
                    )
                binding_key = (
                    binding.instance_ref,
                    binding.profile.stable_id(),
                )
                if binding_key in binding_keys:
                    raise SchemaError(
                        "duplicate instance/profile binding",
                        path=binding_path,
                    )
                binding_keys.add(binding_key)
                profiles_by_instance.setdefault(binding.instance_ref, set()).add(
                    binding.profile
                )
            if set(profiles_by_instance) != set(instance_index):
                raise SchemaError(
                    "must bind every instance at least once",
                    path=f"{path}.instance_profiles",
                )
            if self.profile != self.instance_profiles[0].profile:
                raise SchemaError(
                    "must equal the first canonical instance profile",
                    path=f"{path}.profile",
                )
        else:
            if self.pd_plan_id is not None:
                raise SchemaError(
                    "must be null without instance_profiles",
                    path=f"{path}.pd_plan_id",
                )
            if len(self.instances) != 1:
                raise SchemaError(
                    "multi-instance IR0 requires instance_profiles",
                    path=f"{path}.instance_profiles",
                )
            profiles_by_instance = {self.instances[0].id: {self.profile}}
        if type(self.node_profiles) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.node_profiles"
            )
        node_profile_index: dict[str, ProfileKey] = {}
        for index, binding in enumerate(self.node_profiles):
            binding_path = f"{path}.node_profiles[{index}]"
            if type(binding) is not NodeProfileBinding:
                raise SchemaError("must be a NodeProfileBinding", path=binding_path)
            binding.validate(binding_path)
            if binding.node_ref in node_profile_index:
                raise SchemaError(
                    "duplicate node profile binding",
                    path=f"{binding_path}.node_ref",
                )
            node_profile_index[binding.node_ref] = binding.profile
        multiple_profiles = any(
            len(profiles) > 1 for profiles in profiles_by_instance.values()
        )
        if multiple_profiles:
            if tuple(binding.node_ref for binding in self.node_profiles) != tuple(
                node.id for node in self.nodes
            ):
                raise SchemaError(
                    "multi-profile instances require one binding per node in node order",
                    path=f"{path}.node_profiles",
                )
        elif self.node_profiles:
            raise SchemaError(
                "must be empty when every instance has one profile",
                path=f"{path}.node_profiles",
            )
        all_mesh_ids: set[str] = set()
        meshes_by_instance: dict[str, set[str]] = {}
        mesh_index: dict[str, DeviceMesh] = {}
        for index, instance in enumerate(self.instances):
            instance.validate(f"{path}.instances[{index}]")
            meshes = {mesh.id for mesh in instance.meshes}
            overlap = all_mesh_ids.intersection(meshes)
            if overlap:
                raise SchemaError(
                    f"mesh id {sorted(overlap)[0]!r} is not globally unique",
                    path=f"{path}.instances[{index}].meshes",
                )
            all_mesh_ids.update(meshes)
            meshes_by_instance[instance.id] = meshes
            mesh_index.update({mesh.id: mesh for mesh in instance.meshes})
        node_index, value_index = validate_value_graph(
            self.nodes, self.values, self.edges, path=path
        )
        for index, node in enumerate(self.nodes):
            if node.instance_id not in instance_index:
                raise SchemaError(
                    f"dangling instance {node.instance_id!r}", path=f"{path}.nodes[{index}].instance_id"
                )
            if node.mesh_ref not in meshes_by_instance[node.instance_id]:
                raise SchemaError(
                    f"mesh {node.mesh_ref!r} does not belong to instance {node.instance_id!r}",
                    path=f"{path}.nodes[{index}].mesh_ref",
                )
            expected_profiles = profiles_by_instance[node.instance_id]
            node_profile = node_profile_index.get(node.id)
            if multiple_profiles:
                assert node_profile is not None
                if node_profile not in expected_profiles:
                    raise SchemaError(
                        "node profile must belong to its instance",
                        path=f"{path}.node_profiles[{index}].profile",
                    )
            else:
                node_profile = next(iter(expected_profiles))
            workload_profile = getattr(node.workload, "profile", None)
            if (
                multiple_profiles
                and
                node.kind is not OpKind.ATTENTION
                and workload_profile is not None
                and workload_profile != node_profile
            ):
                raise SchemaError(
                    "workload profile must equal the node profile",
                    path=f"{path}.nodes[{index}].workload.profile",
                )
            if node.kind is OpKind.ATTENTION:
                assert isinstance(node.workload, AttentionWorkload)
                if node.workload.mode is AttentionMode.TRAIN_FORWARD:
                    if node.effects != NodeEffects(EffectKind.PURE, None, None):
                        raise SchemaError(
                            "train-forward attention must be pure and carry no KV effect",
                            path=f"{path}.nodes[{index}].effects",
                        )
                elif (
                    node.effects.kind is not EffectKind.STATEFUL
                    or node.effects.effect_token is None
                    or node.effects.alias_set is None
                ):
                    raise SchemaError(
                        "attention must explicitly model stateful KV-cache effects",
                        path=f"{path}.nodes[{index}].effects",
                    )
                if node.workload.profile != node_profile:
                    raise SchemaError(
                        "attention workload profile must belong to its instance",
                        path=f"{path}.nodes[{index}].workload.profile",
                    )
            if node.kind is OpKind.COLLECTIVE:
                assert isinstance(node.workload, CollectiveWorkload)
                mesh = mesh_index[node.mesh_ref]
                axis_sizes = {axis.name: axis.size for axis in mesh.axes}
                missing_axes = set(node.workload.mesh_axes) - set(axis_sizes)
                if missing_axes:
                    raise SchemaError(
                        "collective mesh axes are absent from the referenced mesh",
                        path=f"{path}.nodes[{index}].workload.mesh_axes",
                    )
                expected_participants = 1
                for axis in node.workload.mesh_axes:
                    expected_participants *= axis_sizes[axis]
                if node.workload.participant_count != expected_participants:
                    raise SchemaError(
                        "participant_count must equal the product of collective mesh axes",
                        path=f"{path}.nodes[{index}].workload.participant_count",
                    )
        if self.persistent_states != tuple(
            sorted(
                self.persistent_states,
                key=lambda item: (item.identity.id, item.id),
            )
        ):
            raise SchemaError(
                "must use canonical identity/id order",
                path=f"{path}.persistent_states",
            )
        if self.state_accesses != tuple(
            sorted(
                self.state_accesses,
                key=lambda item: (
                    item.node_ref, item.state_ref, item.rank, item.id
                ),
            )
        ):
            raise SchemaError(
                "must use canonical node/state/rank/id order",
                path=f"{path}.state_accesses",
            )

        state_index: dict[str, PersistentStateDecl] = {}
        identity_ids: set[str] = set()
        for index, declaration in enumerate(self.persistent_states):
            state_path = f"{path}.persistent_states[{index}]"
            if type(declaration) is not PersistentStateDecl:
                raise SchemaError("must be a PersistentStateDecl", path=state_path)
            declaration.validate(state_path)
            if declaration.id in state_index:
                raise SchemaError(
                    "duplicate state declaration id", path=f"{state_path}.id"
                )
            if declaration.identity.id in identity_ids:
                raise SchemaError(
                    "duplicate persistent state identity",
                    path=f"{state_path}.identity.id",
                )
            instance = instance_index.get(declaration.identity.instance_ref)
            if instance is None:
                raise SchemaError(
                    "state identity references a dangling instance",
                    path=f"{state_path}.identity.instance_ref",
                )
            if declaration.identity.mesh_ref not in meshes_by_instance[instance.id]:
                raise SchemaError(
                    "state identity references a mesh outside its instance",
                    path=f"{state_path}.identity.mesh_ref",
                )
            if declaration.identity.shard_index >= instance.parallel.tp:
                raise SchemaError(
                    "state shard_index must be smaller than instance TP",
                    path=f"{state_path}.identity.shard_index",
                )
            tensor_ref = declaration.identity.tensor_ref
            if tensor_ref is not None and tensor_ref not in value_index:
                raise SchemaError(
                    "state identity references a dangling tensor value",
                    path=f"{state_path}.identity.tensor_ref",
                )
            state_index[declaration.id] = declaration
            identity_ids.add(declaration.identity.id)

        validate_unique_ids(self.state_accesses, f"{path}.state_accesses")
        access_keys: set[tuple[str, str, int]] = set()
        allowed_modes = {
            PersistentStateAccess.READ_ONLY: {StateAccessMode.READ},
            PersistentStateAccess.READ_WRITE: {
                StateAccessMode.READ,
                StateAccessMode.WRITE,
                StateAccessMode.READ_WRITE,
            },
            PersistentStateAccess.RESERVED: set(),
        }
        for index, access in enumerate(self.state_accesses):
            access_path = f"{path}.state_accesses[{index}]"
            if type(access) is not StateAccess:
                raise SchemaError("must be a StateAccess", path=access_path)
            access.validate(access_path)
            node = node_index.get(access.node_ref)
            if node is None:
                raise SchemaError(
                    "state access references a dangling node",
                    path=f"{access_path}.node_ref",
                )
            declaration = state_index.get(access.state_ref)
            if declaration is None:
                raise SchemaError(
                    "state access references a dangling state declaration",
                    path=f"{access_path}.state_ref",
                )
            if node.instance_id != declaration.identity.instance_ref:
                raise SchemaError(
                    "node and persistent state must belong to the same instance",
                    path=access_path,
                )
            if node.mesh_ref != declaration.identity.mesh_ref:
                raise SchemaError(
                    "node and persistent state must use the same mesh",
                    path=access_path,
                )
            if access.rank != declaration.identity.shard_index:
                raise SchemaError(
                    "access rank must equal the state shard_index",
                    path=f"{access_path}.rank",
                )
            if access.mode not in allowed_modes[declaration.access]:
                raise SchemaError(
                    "access mode exceeds declaration permission",
                    path=f"{access_path}.mode",
                )
            if access.mode in (StateAccessMode.READ, StateAccessMode.READ_WRITE):
                state_access_tensor_view(
                    access,
                    declaration,
                    "read",
                    path=access_path,
                )
            if access.mode in (StateAccessMode.WRITE, StateAccessMode.READ_WRITE):
                state_access_tensor_view(
                    access,
                    declaration,
                    "write",
                    path=access_path,
                )
            key = (access.node_ref, access.state_ref, access.rank)
            if key in access_keys:
                raise SchemaError("duplicate logical state access", path=access_path)
            access_keys.add(key)
        for index, value in enumerate(self.values):
            if value.sharding.mesh_ref not in all_mesh_ids:
                raise SchemaError(
                    f"dangling mesh {value.sharding.mesh_ref!r}",
                    path=f"{path}.values[{index}].sharding.mesh_ref",
                )
        validate_unique_ids(self.fusion_candidates, f"{path}.fusion_candidates")
        for index, candidate in enumerate(self.fusion_candidates):
            candidate.validate(f"{path}.fusion_candidates[{index}]")
            member_set = set(candidate.members)
            if not member_set.issubset(node_index):
                raise SchemaError("contains a dangling member", path=f"{path}.fusion_candidates[{index}].members")
            if not set(candidate.boundary_inputs).issubset(value_index):
                raise SchemaError("contains a dangling value", path=f"{path}.fusion_candidates[{index}].boundary_inputs")
            if not set(candidate.boundary_outputs).issubset(value_index):
                raise SchemaError("contains a dangling value", path=f"{path}.fusion_candidates[{index}].boundary_outputs")
            expected_inputs = {
                value.id
                for value in self.values
                if member_set.intersection(value.consumers)
                and (value.producer is None or value.producer not in member_set)
            }
            expected_outputs = {
                value.id
                for value in self.values
                if value.producer in member_set
                and (not value.consumers or any(c not in member_set for c in value.consumers))
            }
            if set(candidate.boundary_inputs) != expected_inputs:
                raise SchemaError("does not match graph-derived boundary inputs", path=f"{path}.fusion_candidates[{index}].boundary_inputs")
            if set(candidate.boundary_outputs) != expected_outputs:
                raise SchemaError("does not match graph-derived boundary outputs", path=f"{path}.fusion_candidates[{index}].boundary_outputs")
        self.profile.validate(f"{path}.profile")
        if self.job is JobKind.TRAIN:
            if self.train is None:
                raise SchemaError("is required for a train job", path=f"{path}.train")
            self.train.validate(f"{path}.train")
        elif self.train is not None:
            raise SchemaError("must be null for an infer job", path=f"{path}.train")
