"""Immutable typed carriers for inter-die Swizzle planning.

The module is deliberately independent from compiler passes.  It records a
fully described mathematical problem, candidate action witnesses, analytical
costs, and the deterministic decision.  Producers must use the ``create``
constructors so identity is derived from semantic content rather than process
state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    DType,
    MeshAxisName,
    stable_artifact_id,
    validate_dependency_dag,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .ir0 import (
    CollectiveKind,
    FusionPattern,
    GemmPartition,
    ReduceOp,
)
from .serde import canonical_digest


SWIZZLE_PROBLEM_SCHEMA_VERSION = "wafer_frontend.swizzle_problem/v1alpha1"
SWIZZLE_ACTION_SCHEMA_VERSION = "wafer_frontend.swizzle_action/v1alpha1"
SWIZZLE_COST_SCHEMA_VERSION = "wafer_frontend.swizzle_cost/v1alpha1"
SWIZZLE_CANDIDATE_SCHEMA_VERSION = "wafer_frontend.swizzle_candidate/v1alpha1"
SWIZZLE_DECISION_SCHEMA_VERSION = "wafer_frontend.swizzle_decision/v1alpha1"
SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_hardware_profile/v1alpha1"
)


class SwizzleAlgorithm(str, Enum):
    UNFUSED = "unfused"
    WANG_1D_BIDIRECTIONAL = "wang_1d_bidirectional"
    MESHSLICE_2D_OS = "meshslice_2d_os"
    DIRECT_XY_PERSONALIZED_A2A = "direct_xy_personalized_a2a"
    COMET_MESH_PERSONALIZED_A2A = "comet_mesh_personalized_a2a"


class SwizzleTensorAxisRole(str, Enum):
    BATCH = "batch"
    FREE_LHS = "free_lhs"
    FREE_RHS = "free_rhs"
    CONTRACT = "contract"


class SwizzleOperand(str, Enum):
    LHS = "lhs"
    RHS = "rhs"
    OUTPUT = "output"


class SwizzleCollectivePosition(str, Enum):
    BEFORE_GEMM = "before_gemm"
    AFTER_GEMM = "after_gemm"


class SwizzleUpdateKind(str, Enum):
    OUTPUT_SLICE = "output_slice"
    PARTIAL_ACCUMULATION = "partial_accumulation"
    REDUCE_THEN_REPLICATE = "reduce_then_replicate"


class SwizzleDecisionReason(str, Enum):
    BASELINE_ONLY = "baseline_only"
    NO_PROFITABLE_FUSION = "no_profitable_fusion"
    LOWEST_ESTIMATED_CYCLES = "lowest_estimated_cycles"
    NON_OVERLAPPING_INTERVAL = "non_overlapping_interval"
    INTERVAL_TIE_BREAK = "interval_tie_break"


class SwizzleActionKind(str, Enum):
    COMP = "comp"
    SWIGLU = "swiglu"
    SEND = "send"
    RECV = "recv"
    WAIT = "wait"
    REDUCE = "reduce"
    LOCAL_COPY = "local_copy"
    BARRIER = "barrier"


class SwizzlePhase(str, Enum):
    PROLOGUE = "prologue"
    STEADY = "steady"
    EPILOGUE = "epilogue"


class SwizzleTopologyKind(str, Enum):
    UNFUSED = "unfused"
    BIDIRECTIONAL_LINE = "bidirectional_line"
    HAMILTONIAN_RING = "hamiltonian_ring"
    RECTANGLE_2D = "rectangle_2d"


def _validate_positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _validate_nonnegative_float(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise SchemaError("must be a finite non-negative float", path=path)


def _validate_fraction(value: float, path: str) -> None:
    _validate_nonnegative_float(value, path)
    if value > 1.0:
        raise SchemaError("must not exceed one", path=path)


def _validate_refs(refs: tuple[str, ...], path: str, *, nonempty: bool = False) -> None:
    if type(refs) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    if nonempty and not refs:
        raise SchemaError("must not be empty", path=path)
    if len(set(refs)) != len(refs):
        raise SchemaError("contains duplicate references", path=path)
    for index, ref in enumerate(refs):
        validate_nonempty(ref, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleTensorAxis:
    tensor_ref: str
    index: int
    name: str
    extent: int
    role: SwizzleTensorAxisRole

    def validate(self, path: str = "swizzle_tensor_axis") -> None:
        validate_nonempty(self.tensor_ref, f"{path}.tensor_ref")
        validate_uint64(self.index, f"{path}.index")
        validate_nonempty(self.name, f"{path}.name")
        _validate_positive(self.extent, f"{path}.extent")
        if type(self.role) is not SwizzleTensorAxisRole:
            raise SchemaError("must be a SwizzleTensorAxisRole", path=f"{path}.role")


@dataclass(frozen=True, slots=True)
class SwizzleTensorView:
    value_ref: str
    shape: tuple[int, ...]
    layout: str
    axis_roles: tuple[SwizzleTensorAxisRole, ...]
    sharding_dim_map: tuple[MeshAxisName | None, ...]
    partial_mesh_axes: tuple[MeshAxisName, ...]

    def validate(self, path: str = "swizzle_tensor_view") -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        validate_nonempty(self.layout, f"{path}.layout")
        if type(self.shape) is not tuple or not self.shape:
            raise SchemaError("must have a non-empty immutable shape", path=f"{path}.shape")
        for index, extent in enumerate(self.shape):
            _validate_positive(extent, f"{path}.shape[{index}]")
        if len(self.axis_roles) != len(self.shape):
            raise SchemaError("must equal tensor rank", path=f"{path}.axis_roles")
        for index, role in enumerate(self.axis_roles):
            if type(role) is not SwizzleTensorAxisRole:
                raise SchemaError("must be a SwizzleTensorAxisRole", path=f"{path}.axis_roles[{index}]")
        if len(self.sharding_dim_map) != len(self.shape):
            raise SchemaError("must equal tensor rank", path=f"{path}.sharding_dim_map")
        mapped = tuple(axis for axis in self.sharding_dim_map if axis is not None)
        if len(set(mapped)) != len(mapped):
            raise SchemaError("maps a mesh axis more than once", path=f"{path}.sharding_dim_map")
        if len(set(self.partial_mesh_axes)) != len(self.partial_mesh_axes):
            raise SchemaError("contains duplicate mesh axes", path=f"{path}.partial_mesh_axes")
        if set(mapped).intersection(self.partial_mesh_axes):
            raise SchemaError("mesh axis cannot be both sharded and partial", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleGemmDescriptor:
    node_ref: str
    partition: GemmPartition
    m: int
    n: int
    k: int
    batch_shape: tuple[int, ...]
    lhs: SwizzleTensorView
    rhs: SwizzleTensorView
    output: SwizzleTensorView
    boundary_input_refs: tuple[str, ...]
    local_operand_refs: tuple[str, ...]
    dtype: DType
    accumulation_dtype: DType
    flops: int

    def validate(self, path: str = "swizzle_gemm") -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.partition) is not GemmPartition:
            raise SchemaError("must be a GemmPartition", path=f"{path}.partition")
        for name in ("m", "n", "k"):
            _validate_positive(getattr(self, name), f"{path}.{name}")
        if type(self.batch_shape) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.batch_shape")
        batch_product = 1
        for index, extent in enumerate(self.batch_shape):
            _validate_positive(extent, f"{path}.batch_shape[{index}]")
            batch_product *= extent
        for name in ("lhs", "rhs", "output"):
            view = getattr(self, name)
            if type(view) is not SwizzleTensorView:
                raise SchemaError("must be a SwizzleTensorView", path=f"{path}.{name}")
            view.validate(f"{path}.{name}")
        if len({self.lhs.value_ref, self.rhs.value_ref, self.output.value_ref}) != 3:
            raise SchemaError("GEMM boundary values must be distinct", path=path)
        _validate_refs(
            self.boundary_input_refs,
            f"{path}.boundary_input_refs",
            nonempty=True,
        )
        _validate_refs(self.local_operand_refs, f"{path}.local_operand_refs")
        if set(self.boundary_input_refs).intersection(self.local_operand_refs):
            raise SchemaError("boundary and local operands must be disjoint", path=path)
        direct_operands = {self.lhs.value_ref, self.rhs.value_ref}
        if not set(self.local_operand_refs).issubset(direct_operands):
            raise SchemaError(
                "local operands must name direct GEMM inputs",
                path=f"{path}.local_operand_refs",
            )
        if self.output.value_ref in set(self.boundary_input_refs):
            raise SchemaError("GEMM output cannot be a boundary input", path=path)
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("must use typed dtypes", path=f"{path}.dtype")
        _validate_positive(self.flops, f"{path}.flops")
        expected_flops = 2 * batch_product * self.m * self.n * self.k
        if self.flops != expected_flops:
            raise SchemaError(
                f"must equal logical GEMM FLOPs {expected_flops}", path=f"{path}.flops"
            )
        expected_batch_roles = (SwizzleTensorAxisRole.BATCH,) * len(self.batch_shape)
        for name, expected_roles in (
            ("lhs", expected_batch_roles + (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.CONTRACT)),
            ("rhs", expected_batch_roles + (SwizzleTensorAxisRole.CONTRACT, SwizzleTensorAxisRole.FREE_RHS)),
            ("output", expected_batch_roles + (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.FREE_RHS)),
        ):
            view = getattr(self, name)
            if view.axis_roles != expected_roles:
                raise SchemaError("axis roles do not match GEMM semantics", path=f"{path}.{name}.axis_roles")
        expected_shapes = {
            "lhs": self.batch_shape + (self.m, self.k),
            "rhs": self.batch_shape + (self.k, self.n),
            "output": self.batch_shape + (self.m, self.n),
        }
        for name, expected in expected_shapes.items():
            if getattr(self, name).shape != expected:
                raise SchemaError(f"shape must equal {expected!r}", path=f"{path}.{name}.shape")


@dataclass(frozen=True, slots=True)
class SwizzleCollectiveDescriptor:
    node_ref: str
    kind: CollectiveKind
    reduce_op: ReduceOp | None
    position: SwizzleCollectivePosition
    mesh_axes: tuple[MeshAxisName, ...]
    participant_ranks: tuple[int, ...]
    gather_tensor_axis: int | None
    scatter_tensor_axis: int | None
    logical_bytes: int
    rank_input_bytes: int
    rank_output_bytes: int
    input: SwizzleTensorView
    output: SwizzleTensorView

    def validate(self, path: str = "swizzle_collective") -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.kind) is not CollectiveKind:
            raise SchemaError("must be a CollectiveKind", path=f"{path}.kind")
        if type(self.position) is not SwizzleCollectivePosition:
            raise SchemaError("must be a SwizzleCollectivePosition", path=f"{path}.position")
        if not self.mesh_axes or len(set(self.mesh_axes)) != len(self.mesh_axes):
            raise SchemaError("must contain unique mesh axes", path=f"{path}.mesh_axes")
        if type(self.participant_ranks) is not tuple or len(self.participant_ranks) <= 1:
            raise SchemaError("must contain at least two participant ranks", path=f"{path}.participant_ranks")
        if tuple(sorted(set(self.participant_ranks))) != self.participant_ranks:
            raise SchemaError("must contain unique ranks in canonical order", path=f"{path}.participant_ranks")
        for index, rank in enumerate(self.participant_ranks):
            validate_uint64(rank, f"{path}.participant_ranks[{index}]")
        for name in ("gather_tensor_axis", "scatter_tensor_axis"):
            axis = getattr(self, name)
            if axis is not None:
                validate_uint64(axis, f"{path}.{name}")
        for name in ("logical_bytes", "rank_input_bytes", "rank_output_bytes"):
            _validate_positive(getattr(self, name), f"{path}.{name}")
        for name in ("input", "output"):
            view = getattr(self, name)
            if type(view) is not SwizzleTensorView:
                raise SchemaError("must be a SwizzleTensorView", path=f"{path}.{name}")
            view.validate(f"{path}.{name}")
        participants = len(self.participant_ranks)
        if self.kind is CollectiveKind.ALL_GATHER:
            if self.position is not SwizzleCollectivePosition.BEFORE_GEMM:
                raise SchemaError("AllGather fusion must precede GEMM", path=f"{path}.position")
            if self.reduce_op is not None or self.gather_tensor_axis is None or self.scatter_tensor_axis is not None:
                raise SchemaError("AllGather requires only gather_tensor_axis and no reduce_op", path=path)
            axis = self.gather_tensor_axis
            assert axis is not None
            if axis >= len(self.input.shape) or len(self.input.shape) != len(self.output.shape):
                raise SchemaError("gather axis is out of range", path=f"{path}.gather_tensor_axis")
            expected_shape = list(self.input.shape)
            expected_shape[axis] *= participants
            if tuple(expected_shape) != self.output.shape:
                raise SchemaError("output shape must gather participant shards", path=f"{path}.output.shape")
            if self.rank_input_bytes * participants != self.logical_bytes or self.rank_output_bytes != self.logical_bytes:
                raise SchemaError("AllGather byte accounting is inconsistent", path=path)
        elif self.kind is CollectiveKind.REDUCE_SCATTER:
            if self.position is not SwizzleCollectivePosition.AFTER_GEMM:
                raise SchemaError("ReduceScatter fusion must follow GEMM", path=f"{path}.position")
            if self.reduce_op is not ReduceOp.SUM or self.scatter_tensor_axis is None or self.gather_tensor_axis is not None:
                raise SchemaError("ReduceScatter requires SUM and only scatter_tensor_axis", path=path)
            axis = self.scatter_tensor_axis
            assert axis is not None
            if axis >= len(self.input.shape) or len(self.input.shape) != len(self.output.shape):
                raise SchemaError("scatter axis is out of range", path=f"{path}.scatter_tensor_axis")
            expected_shape = list(self.input.shape)
            if expected_shape[axis] % participants:
                raise SchemaError("scatter extent must divide by participants", path=f"{path}.input.shape")
            expected_shape[axis] //= participants
            if tuple(expected_shape) != self.output.shape:
                raise SchemaError("output shape must be one reduced shard", path=f"{path}.output.shape")
            if self.rank_input_bytes != self.logical_bytes or self.rank_output_bytes * participants != self.logical_bytes:
                raise SchemaError("ReduceScatter byte accounting is inconsistent", path=path)
        elif self.kind is CollectiveKind.ALL_REDUCE:
            if self.position is not SwizzleCollectivePosition.AFTER_GEMM:
                raise SchemaError("AllReduce fusion must follow GEMM", path=f"{path}.position")
            if self.reduce_op is not ReduceOp.SUM or self.gather_tensor_axis is not None or self.scatter_tensor_axis is not None:
                raise SchemaError("AllReduce requires SUM and no tensor axis", path=path)
            if self.input.shape != self.output.shape or self.rank_input_bytes != self.logical_bytes or self.rank_output_bytes != self.logical_bytes:
                raise SchemaError("AllReduce shape/byte accounting is inconsistent", path=path)
        else:
            raise SchemaError("unsupported Swizzle collective kind", path=f"{path}.kind")


@dataclass(frozen=True, slots=True)
class SwizzleRankPlacement:
    rank: int
    x: int
    y: int

    def validate(self, path: str = "swizzle_rank_placement") -> None:
        for name in ("rank", "x", "y"):
            validate_uint64(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleRouteView:
    id: str
    source_rank: int
    destination_rank: int
    die_path: tuple[int, ...]
    resource_ids: tuple[str, ...]

    def validate(self, path: str = "swizzle_route") -> None:
        validate_nonempty(self.id, f"{path}.id")
        for name in ("source_rank", "destination_rank"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank:
            raise SchemaError("route endpoints must differ", path=path)
        if type(self.die_path) is not tuple or len(self.die_path) < 2:
            raise SchemaError("must contain at least two dies", path=f"{path}.die_path")
        for index, die in enumerate(self.die_path):
            validate_uint64(die, f"{path}.die_path[{index}]")
        if len(set(self.die_path)) != len(self.die_path):
            raise SchemaError("route must be simple", path=f"{path}.die_path")
        _validate_refs(self.resource_ids, f"{path}.resource_ids", nonempty=True)
        # PairRoute incidence cardinality is intentionally independent of hops.


@dataclass(frozen=True, slots=True)
class SwizzleGroupView:
    group_ref: str
    logical_shape: tuple[int, int]
    placements: tuple[SwizzleRankPlacement, ...]
    routes: tuple[SwizzleRouteView, ...]

    def validate(self, path: str = "swizzle_group") -> None:
        validate_nonempty(self.group_ref, f"{path}.group_ref")
        if type(self.logical_shape) is not tuple or len(self.logical_shape) != 2:
            raise SchemaError("must be a 2D logical shape", path=f"{path}.logical_shape")
        for index, extent in enumerate(self.logical_shape):
            _validate_positive(extent, f"{path}.logical_shape[{index}]")
        if type(self.placements) is not tuple or not self.placements:
            raise SchemaError("must contain placements", path=f"{path}.placements")
        for index, placement in enumerate(self.placements):
            placement.validate(f"{path}.placements[{index}]")
        if tuple(item.rank for item in self.placements) != tuple(range(len(self.placements))):
            raise SchemaError("placements must be canonically ordered dense ranks", path=f"{path}.placements")
        coordinates = tuple((item.x, item.y) for item in self.placements)
        if len(set(coordinates)) != len(coordinates):
            raise SchemaError("contains duplicate coordinates", path=f"{path}.placements")
        if self.logical_shape[0] * self.logical_shape[1] != len(self.placements):
            raise SchemaError("logical shape must cover every rank", path=f"{path}.logical_shape")
        route_index = validate_unique_ids(self.routes, f"{path}.routes")
        del route_index
        ranks = set(range(len(self.placements)))
        pairs: set[tuple[int, int]] = set()
        for index, route in enumerate(self.routes):
            route.validate(f"{path}.routes[{index}]")
            if route.source_rank not in ranks or route.destination_rank not in ranks:
                raise SchemaError("route endpoint is outside the group", path=f"{path}.routes[{index}]")
            pair = (route.source_rank, route.destination_rank)
            if pair in pairs:
                raise SchemaError("contains duplicate endpoint pair", path=f"{path}.routes[{index}]")
            pairs.add(pair)


@dataclass(frozen=True, slots=True)
class SwizzleEfficiencyPoint:
    m: int
    n: int
    k: int
    efficiency: float

    def validate(self, path: str = "swizzle_efficiency_point") -> None:
        for name in ("m", "n", "k"):
            _validate_positive(getattr(self, name), f"{path}.{name}")
        _validate_fraction(self.efficiency, f"{path}.efficiency")
        if self.efficiency == 0.0:
            raise SchemaError("must be greater than zero", path=f"{path}.efficiency")


@dataclass(frozen=True, slots=True)
class SwizzleHardwareProfile:
    schema_version: str
    id: str
    profile_digest: str
    peak_flops_per_cycle: float
    confidence_fraction: float
    efficiency_points: tuple[SwizzleEfficiencyPoint, ...]
    dte_launch_cycles: int
    dte_sync_cycles: int
    hop_latency_cycles: int
    lane_bytes_per_cycle: float
    max_inflight_dte: int
    min_transfer_bytes: int
    efficient_tile_floor: tuple[int, int, int]
    sram_budget_bytes: int
    double_buffer_supported: bool

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleHardwareProfile":
        profile_digest = canonical_digest(semantic)
        result = cls(
            schema_version=SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_hardware_profile", semantic, schema_version=SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION),
            profile_digest=profile_digest,
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "peak_flops_per_cycle", "confidence_fraction", "efficiency_points",
                "dte_launch_cycles", "dte_sync_cycles", "hop_latency_cycles",
                "lane_bytes_per_cycle", "max_inflight_dte", "min_transfer_bytes",
                "efficient_tile_floor", "sram_budget_bytes", "double_buffer_supported",
            )
        }

    def validate(self, path: str = "swizzle_hardware_profile") -> None:
        if self.schema_version != SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        _validate_nonnegative_float(self.peak_flops_per_cycle, f"{path}.peak_flops_per_cycle")
        if self.peak_flops_per_cycle == 0.0:
            raise SchemaError("must be greater than zero", path=f"{path}.peak_flops_per_cycle")
        _validate_fraction(self.confidence_fraction, f"{path}.confidence_fraction")
        if type(self.efficiency_points) is not tuple or not self.efficiency_points:
            raise SchemaError("must contain efficiency points", path=f"{path}.efficiency_points")
        keys: list[tuple[int, int, int]] = []
        for index, point in enumerate(self.efficiency_points):
            point.validate(f"{path}.efficiency_points[{index}]")
            keys.append((point.m, point.n, point.k))
        if tuple(sorted(set(keys))) != tuple(keys):
            raise SchemaError("must be unique and canonically ordered", path=f"{path}.efficiency_points")
        for name in ("dte_launch_cycles", "dte_sync_cycles", "hop_latency_cycles"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        _validate_nonnegative_float(self.lane_bytes_per_cycle, f"{path}.lane_bytes_per_cycle")
        if self.lane_bytes_per_cycle == 0.0:
            raise SchemaError("must be greater than zero", path=f"{path}.lane_bytes_per_cycle")
        for name in ("max_inflight_dte", "min_transfer_bytes", "sram_budget_bytes"):
            _validate_positive(getattr(self, name), f"{path}.{name}")
        if type(self.efficient_tile_floor) is not tuple or len(self.efficient_tile_floor) != 3:
            raise SchemaError("must be an M/N/K triple", path=f"{path}.efficient_tile_floor")
        for index, extent in enumerate(self.efficient_tile_floor):
            _validate_positive(extent, f"{path}.efficient_tile_floor[{index}]")
        if type(self.double_buffer_supported) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.double_buffer_supported")
        semantic = self._semantic_key()
        if self.profile_digest != canonical_digest(semantic):
            raise SchemaError("profile digest does not match semantic content", path=f"{path}.profile_digest")
        expected = stable_artifact_id("swizzle_hardware_profile", semantic, schema_version=SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleConstraints:
    allowed_algorithms: tuple[SwizzleAlgorithm, ...]
    max_candidates: int
    max_actions: int
    max_buffers: int
    max_chunk_count: int
    allow_unroll_two: bool

    def validate(self, path: str = "swizzle_constraints") -> None:
        if type(self.allowed_algorithms) is not tuple or not self.allowed_algorithms:
            raise SchemaError("must contain allowed algorithms", path=f"{path}.allowed_algorithms")
        if len(set(self.allowed_algorithms)) != len(self.allowed_algorithms):
            raise SchemaError("contains duplicate algorithms", path=f"{path}.allowed_algorithms")
        if tuple(sorted(self.allowed_algorithms, key=lambda item: item.value)) != self.allowed_algorithms:
            raise SchemaError("must be canonically ordered", path=f"{path}.allowed_algorithms")
        if SwizzleAlgorithm.UNFUSED not in self.allowed_algorithms:
            raise SchemaError("must retain UNFUSED", path=f"{path}.allowed_algorithms")
        for name in ("max_candidates", "max_actions", "max_buffers", "max_chunk_count"):
            _validate_positive(getattr(self, name), f"{path}.{name}")
        if type(self.allow_unroll_two) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.allow_unroll_two")


@dataclass(frozen=True, slots=True)
class SwizzleProblem:
    schema_version: str
    id: str
    source_ir1_id: str
    fused_op_id: str
    pattern: FusionPattern
    gemm: SwizzleGemmDescriptor
    collective: SwizzleCollectiveDescriptor
    group: SwizzleGroupView
    hardware_profile: SwizzleHardwareProfile
    constraints: SwizzleConstraints

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleProblem":
        result = cls(
            schema_version=SWIZZLE_PROBLEM_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_problem", semantic, schema_version=SWIZZLE_PROBLEM_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "fused_op_id", "pattern", "gemm", "collective",
            "group", "hardware_profile", "constraints",
        )}

    def validate(self, path: str = "swizzle_problem") -> None:
        if self.schema_version != SWIZZLE_PROBLEM_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        validate_nonempty(self.fused_op_id, f"{path}.fused_op_id")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must be a FusionPattern", path=f"{path}.pattern")
        if self.pattern not in (
            FusionPattern.AG_GEMM,
            FusionPattern.GEMM_RS,
            FusionPattern.GEMM_AR,
        ):
            raise SchemaError(
                "MoE pattern requires MoeSwizzleProblem", path=f"{path}.pattern"
            )
        self.gemm.validate(f"{path}.gemm")
        self.collective.validate(f"{path}.collective")
        self.group.validate(f"{path}.group")
        self.hardware_profile.validate(f"{path}.hardware_profile")
        self.constraints.validate(f"{path}.constraints")
        if self.collective.participant_ranks != tuple(range(len(self.group.placements))):
            raise SchemaError("collective participants must exactly cover the group", path=f"{path}.collective.participant_ranks")
        expected_kind = {
            FusionPattern.AG_GEMM: CollectiveKind.ALL_GATHER,
            FusionPattern.GEMM_RS: CollectiveKind.REDUCE_SCATTER,
            FusionPattern.GEMM_AR: CollectiveKind.ALL_REDUCE,
        }[self.pattern]
        if self.collective.kind is not expected_kind:
            raise SchemaError("pattern and collective kind do not match", path=f"{path}.collective.kind")
        expected = stable_artifact_id("swizzle_problem", self._semantic_key(), schema_version=SWIZZLE_PROBLEM_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleSemanticWitness:
    pattern: FusionPattern
    member_refs: tuple[str, str]
    boundary_input_refs: tuple[str, ...]
    boundary_output_refs: tuple[str, ...]
    intermediate_value_ref: str
    gemm_operand: SwizzleOperand
    split_axis: SwizzleTensorAxis
    update_kind: SwizzleUpdateKind
    gather_axis: int | None
    reduction_axis: int | None
    has_reduction_phase: bool
    has_replication_phase: bool
    input_layout_closed: bool
    output_layout_closed: bool
    sharding_transition_closed: bool

    def validate(self, path: str = "swizzle_semantic_witness") -> None:
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must be a FusionPattern", path=f"{path}.pattern")
        _validate_refs(self.member_refs, f"{path}.member_refs", nonempty=True)
        if len(self.member_refs) != 2:
            raise SchemaError("must name exactly two members", path=f"{path}.member_refs")
        _validate_refs(self.boundary_input_refs, f"{path}.boundary_input_refs", nonempty=True)
        _validate_refs(self.boundary_output_refs, f"{path}.boundary_output_refs", nonempty=True)
        validate_nonempty(self.intermediate_value_ref, f"{path}.intermediate_value_ref")
        if self.intermediate_value_ref in set(self.boundary_input_refs + self.boundary_output_refs):
            raise SchemaError("intermediate cannot be a boundary value", path=f"{path}.intermediate_value_ref")
        if type(self.gemm_operand) is not SwizzleOperand:
            raise SchemaError("must be a SwizzleOperand", path=f"{path}.gemm_operand")
        self.split_axis.validate(f"{path}.split_axis")
        if type(self.update_kind) is not SwizzleUpdateKind:
            raise SchemaError("must be a SwizzleUpdateKind", path=f"{path}.update_kind")
        for name in ("gather_axis", "reduction_axis"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        for name in (
            "has_reduction_phase", "has_replication_phase", "input_layout_closed",
            "output_layout_closed", "sharding_transition_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise SchemaError("must be a bool", path=f"{path}.{name}")
        if not (self.input_layout_closed and self.output_layout_closed and self.sharding_transition_closed):
            raise SchemaError("semantic closure witnesses must all pass", path=path)
        if self.pattern is FusionPattern.AG_GEMM:
            expected = (SwizzleUpdateKind.OUTPUT_SLICE, False, False)
            if self.split_axis.role is SwizzleTensorAxisRole.CONTRACT:
                expected = (SwizzleUpdateKind.PARTIAL_ACCUMULATION, False, False)
            if (self.update_kind, self.has_reduction_phase, self.has_replication_phase) != expected or self.gather_axis is None or self.reduction_axis is not None:
                raise SchemaError("invalid AG+GEMM semantic witness", path=path)
        elif self.pattern is FusionPattern.GEMM_RS:
            if (
                self.gemm_operand is not SwizzleOperand.OUTPUT
                or self.update_kind is not SwizzleUpdateKind.PARTIAL_ACCUMULATION
                or self.gather_axis is not None
                or self.reduction_axis is None
                or not self.has_reduction_phase
                or self.has_replication_phase
            ):
                raise SchemaError("invalid GEMM+RS semantic witness", path=path)
        elif self.pattern is FusionPattern.GEMM_AR:
            if (
                self.gemm_operand is not SwizzleOperand.OUTPUT
                or self.update_kind is not SwizzleUpdateKind.REDUCE_THEN_REPLICATE
                or self.gather_axis is not None
                or self.reduction_axis is None
                or not self.has_reduction_phase
                or not self.has_replication_phase
            ):
                raise SchemaError("invalid GEMM+AR two-phase witness", path=path)
        else:
            raise SchemaError(
                "MoE pattern requires MoeSemanticWitness", path=f"{path}.pattern"
            )


@dataclass(frozen=True, slots=True)
class SwizzleTopologyWitness:
    kind: SwizzleTopologyKind
    rank_order: tuple[int, ...]
    row_orders: tuple[tuple[int, ...], ...]
    column_orders: tuple[tuple[int, ...], ...]
    route_refs: tuple[str, ...]
    is_complete_rectangle: bool
    has_hamiltonian_cycle: bool

    def validate(self, path: str = "swizzle_topology_witness") -> None:
        if type(self.kind) is not SwizzleTopologyKind:
            raise SchemaError("must be a SwizzleTopologyKind", path=f"{path}.kind")
        if type(self.rank_order) is not tuple or len(set(self.rank_order)) != len(self.rank_order):
            raise SchemaError("rank order must be immutable and unique", path=f"{path}.rank_order")
        for index, rank in enumerate(self.rank_order):
            validate_uint64(rank, f"{path}.rank_order[{index}]")
        for name in ("row_orders", "column_orders"):
            orders = getattr(self, name)
            if type(orders) is not tuple:
                raise SchemaError("must be an immutable tuple", path=f"{path}.{name}")
            for index, order in enumerate(orders):
                if not order or len(set(order)) != len(order):
                    raise SchemaError("each order must be non-empty and unique", path=f"{path}.{name}[{index}]")
        _validate_refs(self.route_refs, f"{path}.route_refs")
        if type(self.is_complete_rectangle) is not bool or type(self.has_hamiltonian_cycle) is not bool:
            raise SchemaError("topology predicates must be bools", path=path)
        if self.kind is SwizzleTopologyKind.HAMILTONIAN_RING and not self.has_hamiltonian_cycle:
            raise SchemaError("ring requires a Hamiltonian cycle witness", path=path)
        if self.kind is SwizzleTopologyKind.RECTANGLE_2D and not self.is_complete_rectangle:
            raise SchemaError("2D topology requires a complete rectangle", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleFeasibilityCheck:
    name: str
    passed: bool
    reason: str

    def validate(self, path: str = "swizzle_feasibility_check") -> None:
        validate_nonempty(self.name, f"{path}.name")
        if type(self.passed) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.passed")
        validate_nonempty(self.reason, f"{path}.reason")


@dataclass(frozen=True, slots=True)
class SwizzleFeasibilityWitness:
    checks: tuple[SwizzleFeasibilityCheck, ...]

    def validate(self, path: str = "swizzle_feasibility_witness") -> None:
        if type(self.checks) is not tuple or not self.checks:
            raise SchemaError("must contain checks", path=f"{path}.checks")
        names: list[str] = []
        for index, check in enumerate(self.checks):
            check.validate(f"{path}.checks[{index}]")
            names.append(check.name)
        if tuple(sorted(set(names))) != tuple(names):
            raise SchemaError("checks must be unique and canonically ordered", path=f"{path}.checks")

    @property
    def feasible(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(frozen=True, slots=True)
class SwizzleBufferRequirement:
    rank: int
    buffer_ref: str
    size_bytes: int
    double_buffered: bool
    lifetime_action_refs: tuple[str, ...]

    def validate(self, path: str = "swizzle_buffer_requirement") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        _validate_positive(self.size_bytes, f"{path}.size_bytes")
        if type(self.double_buffered) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.double_buffered")
        _validate_refs(self.lifetime_action_refs, f"{path}.lifetime_action_refs", nonempty=True)


@dataclass(frozen=True, slots=True)
class SwizzleActionWitness:
    schema_version: str
    id: str
    rank: int
    kind: SwizzleActionKind
    deps: tuple[str, ...]
    chunk_index: int | None
    phase: SwizzlePhase
    peer_rank: int | None
    route_ref: str | None
    input_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    logical_bytes: int
    flops: int

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleActionWitness":
        result = cls(
            schema_version=SWIZZLE_ACTION_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_action", semantic, schema_version=SWIZZLE_ACTION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "rank", "kind", "deps", "chunk_index", "phase", "peer_rank",
            "route_ref", "input_refs", "output_refs", "logical_bytes", "flops",
        )}

    def validate(self, path: str = "swizzle_action") -> None:
        if self.schema_version != SWIZZLE_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.kind) is not SwizzleActionKind:
            raise SchemaError("must be a SwizzleActionKind", path=f"{path}.kind")
        _validate_refs(self.deps, f"{path}.deps")
        if self.chunk_index is not None:
            validate_uint64(self.chunk_index, f"{path}.chunk_index")
        if type(self.phase) is not SwizzlePhase:
            raise SchemaError("must be a SwizzlePhase", path=f"{path}.phase")
        if self.peer_rank is not None:
            validate_uint64(self.peer_rank, f"{path}.peer_rank")
            if self.peer_rank == self.rank:
                raise SchemaError("peer rank must differ", path=f"{path}.peer_rank")
        if self.route_ref is not None:
            validate_nonempty(self.route_ref, f"{path}.route_ref")
        _validate_refs(self.input_refs, f"{path}.input_refs")
        _validate_refs(self.output_refs, f"{path}.output_refs")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        if self.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            if self.peer_rank is None or self.route_ref is None or self.logical_bytes == 0 or self.flops != 0:
                raise SchemaError("transport requires peer/route/bytes and zero FLOPs", path=path)
        elif self.kind is SwizzleActionKind.COMP:
            if self.peer_rank is not None or self.route_ref is not None or self.flops == 0 or self.logical_bytes != 0:
                raise SchemaError("compute requires FLOPs and no transport fields", path=path)
        elif self.peer_rank is not None or self.route_ref is not None:
            raise SchemaError("non-transport action cannot carry peer/route", path=path)
        expected = stable_artifact_id("swizzle_action", self._semantic_key(), schema_version=SWIZZLE_ACTION_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleRankProgramWitness:
    rank: int
    actions: tuple[SwizzleActionWitness, ...]

    def validate(self, path: str = "swizzle_rank_program") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("must contain actions", path=f"{path}.actions")
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.rank != self.rank:
                raise SchemaError("action rank must equal program rank", path=f"{path}.actions[{index}].rank")


@dataclass(frozen=True, slots=True)
class SwizzleCost:
    schema_version: str
    id: str
    estimated_cycles: float
    lower_cycles: float
    upper_cycles: float
    prologue_cycles: float
    steady_cycles: float
    epilogue_cycles: float
    logical_bytes: int
    byte_hops: int
    message_count: int
    direction_port_utilization: float
    control_action_count: int
    max_inflight: int
    sram_high_water_bytes: int
    bottleneck_resources: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleCost":
        result = cls(
            schema_version=SWIZZLE_COST_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_cost", semantic, schema_version=SWIZZLE_COST_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "estimated_cycles", "lower_cycles", "upper_cycles", "prologue_cycles",
            "steady_cycles", "epilogue_cycles", "logical_bytes", "byte_hops",
            "message_count", "direction_port_utilization", "control_action_count",
            "max_inflight", "sram_high_water_bytes", "bottleneck_resources",
        )}

    def validate(self, path: str = "swizzle_cost") -> None:
        if self.schema_version != SWIZZLE_COST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("estimated_cycles", "lower_cycles", "upper_cycles", "prologue_cycles", "steady_cycles", "epilogue_cycles"):
            _validate_nonnegative_float(getattr(self, name), f"{path}.{name}")
        if not self.lower_cycles <= self.estimated_cycles <= self.upper_cycles:
            raise SchemaError("estimate must lie in confidence interval", path=f"{path}.estimated_cycles")
        if not math.isclose(self.estimated_cycles, self.prologue_cycles + self.steady_cycles + self.epilogue_cycles, rel_tol=1e-9, abs_tol=1e-9):
            raise SchemaError("phase cycles must sum to estimate", path=f"{path}.estimated_cycles")
        for name in ("logical_bytes", "byte_hops", "message_count", "control_action_count", "max_inflight", "sram_high_water_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        _validate_fraction(self.direction_port_utilization, f"{path}.direction_port_utilization")
        _validate_refs(self.bottleneck_resources, f"{path}.bottleneck_resources")
        if tuple(sorted(self.bottleneck_resources)) != self.bottleneck_resources:
            raise SchemaError("must be canonically ordered", path=f"{path}.bottleneck_resources")
        expected = stable_artifact_id("swizzle_cost", self._semantic_key(), schema_version=SWIZZLE_COST_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleCandidate:
    schema_version: str
    id: str
    problem_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    split_axis: SwizzleTensorAxis | None
    chunk_count: int
    unroll_degree: int
    rank_programs: tuple[SwizzleRankProgramWitness, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]
    topology_witness: SwizzleTopologyWitness
    semantic_witness: SwizzleSemanticWitness
    feasibility_witness: SwizzleFeasibilityWitness
    cost: SwizzleCost

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleCandidate":
        result = cls(
            schema_version=SWIZZLE_CANDIDATE_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_candidate", semantic, schema_version=SWIZZLE_CANDIDATE_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "problem_ref", "pattern", "algorithm", "split_axis", "chunk_count",
            "unroll_degree", "rank_programs", "buffer_requirements",
            "topology_witness", "semantic_witness", "feasibility_witness", "cost",
        )}

    def validate(self, path: str = "swizzle_candidate") -> None:
        if self.schema_version != SWIZZLE_CANDIDATE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.problem_ref, f"{path}.problem_ref")
        if type(self.pattern) is not FusionPattern or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must use typed pattern and algorithm", path=path)
        validate_uint64(self.chunk_count, f"{path}.chunk_count")
        validate_uint64(self.unroll_degree, f"{path}.unroll_degree")
        self.topology_witness.validate(f"{path}.topology_witness")
        self.semantic_witness.validate(f"{path}.semantic_witness")
        self.feasibility_witness.validate(f"{path}.feasibility_witness")
        self.cost.validate(f"{path}.cost")
        if self.semantic_witness.pattern is not self.pattern:
            raise SchemaError("semantic witness pattern mismatch", path=f"{path}.semantic_witness.pattern")
        if self.algorithm is SwizzleAlgorithm.UNFUSED:
            if self.split_axis is not None or self.chunk_count != 0 or self.unroll_degree != 0 or self.rank_programs or self.buffer_requirements or self.topology_witness.kind is not SwizzleTopologyKind.UNFUSED:
                raise SchemaError("UNFUSED must not carry decomposed execution", path=path)
        else:
            if self.split_axis is None or self.chunk_count == 0 or self.unroll_degree not in (1, 2) or not self.rank_programs:
                raise SchemaError("fused candidate requires split/action program", path=path)
            self.split_axis.validate(f"{path}.split_axis")
            if self.split_axis != self.semantic_witness.split_axis:
                raise SchemaError("split axis must equal semantic witness", path=f"{path}.split_axis")
            if not self.feasibility_witness.feasible:
                raise SchemaError("materialized candidate must be feasible", path=f"{path}.feasibility_witness")
        if tuple(program.rank for program in self.rank_programs) != tuple(range(len(self.rank_programs))):
            raise SchemaError("rank programs must be canonically ordered dense ranks", path=f"{path}.rank_programs")
        all_actions: tuple[SwizzleActionWitness, ...] = ()
        for index, program in enumerate(self.rank_programs):
            program.validate(f"{path}.rank_programs[{index}]")
            all_actions += program.actions
        if all_actions:
            action_index = validate_dependency_dag(all_actions, f"{path}.rank_programs.actions")
            route_refs = set(self.topology_witness.route_refs)
            for action in all_actions:
                if action.route_ref is not None and action.route_ref not in route_refs:
                    raise SchemaError("action references a route outside topology witness", path=f"{path}.rank_programs")
            buffer_keys: set[tuple[int, str]] = set()
            for index, requirement in enumerate(self.buffer_requirements):
                requirement.validate(f"{path}.buffer_requirements[{index}]")
                key = (requirement.rank, requirement.buffer_ref)
                if key in buffer_keys:
                    raise SchemaError("duplicate rank/buffer requirement", path=f"{path}.buffer_requirements[{index}]")
                buffer_keys.add(key)
                for ref in requirement.lifetime_action_refs:
                    if ref not in action_index:
                        raise SchemaError("buffer lifetime references an unknown action", path=f"{path}.buffer_requirements[{index}].lifetime_action_refs")
        expected = stable_artifact_id("swizzle_candidate", self._semantic_key(), schema_version=SWIZZLE_CANDIDATE_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleDecision:
    schema_version: str
    id: str
    problem: SwizzleProblem
    baseline: SwizzleCandidate
    ranked_candidates: tuple[SwizzleCandidate, ...]
    selected_candidate_ref: str
    decision_reason: SwizzleDecisionReason

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleDecision":
        result = cls(
            schema_version=SWIZZLE_DECISION_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_decision", semantic, schema_version=SWIZZLE_DECISION_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "problem", "baseline", "ranked_candidates", "selected_candidate_ref",
            "decision_reason",
        )}

    def validate(self, path: str = "swizzle_decision") -> None:
        if self.schema_version != SWIZZLE_DECISION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.problem.validate(f"{path}.problem")
        self.baseline.validate(f"{path}.baseline")
        if self.baseline.algorithm is not SwizzleAlgorithm.UNFUSED:
            raise SchemaError("baseline must be UNFUSED", path=f"{path}.baseline.algorithm")
        if type(self.ranked_candidates) is not tuple or not self.ranked_candidates:
            raise SchemaError("must retain a ranked candidate set", path=f"{path}.ranked_candidates")
        candidates = validate_unique_ids(self.ranked_candidates, f"{path}.ranked_candidates")
        if self.ranked_candidates[0].id != self.selected_candidate_ref:
            raise SchemaError("selected candidate must be ranked first", path=f"{path}.selected_candidate_ref")
        for index, candidate in enumerate(self.ranked_candidates):
            candidate.validate(f"{path}.ranked_candidates[{index}]")
            if candidate.problem_ref != self.problem.id or candidate.pattern is not self.problem.pattern:
                raise SchemaError("candidate does not belong to problem", path=f"{path}.ranked_candidates[{index}]")
        if self.baseline.problem_ref != self.problem.id or self.baseline.pattern is not self.problem.pattern:
            raise SchemaError("baseline does not belong to problem", path=f"{path}.baseline")
        if self.baseline.id not in candidates:
            raise SchemaError("ranked candidates must include baseline", path=f"{path}.ranked_candidates")
        if type(self.decision_reason) is not SwizzleDecisionReason:
            raise SchemaError("must be a SwizzleDecisionReason", path=f"{path}.decision_reason")
        selected = candidates[self.selected_candidate_ref]
        if selected.algorithm is SwizzleAlgorithm.UNFUSED and self.decision_reason not in (
            SwizzleDecisionReason.BASELINE_ONLY, SwizzleDecisionReason.NO_PROFITABLE_FUSION,
        ):
            raise SchemaError("baseline selection requires a baseline reason", path=f"{path}.decision_reason")
        if selected.algorithm is not SwizzleAlgorithm.UNFUSED and self.decision_reason in (
            SwizzleDecisionReason.BASELINE_ONLY, SwizzleDecisionReason.NO_PROFITABLE_FUSION,
        ):
            raise SchemaError("fused selection requires a fusion reason", path=f"{path}.decision_reason")
        expected = stable_artifact_id("swizzle_decision", self._semantic_key(), schema_version=SWIZZLE_DECISION_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("Swizzle") or name.startswith("SWIZZLE_")] + ["FusionPattern"]
