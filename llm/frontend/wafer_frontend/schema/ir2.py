"""Semantic per-die DAGs and their separately versioned physical schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..errors import SchemaError
from .action import (
    BarrierContract,
    BarrierScope,
    canonical_compute_operand_roles,
    ComputeContract,
    ComputeOperand,
    FusionAction,
    FusionActionKind,
    FusionPlan,
    ReductionContract,
    StandaloneCollectivePlan,
    SyncContract,
)
from .common import (
    DType,
    Sharding,
    UINT64_MAX,
    stable_artifact_id,
    validate_dependency_dag,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .ir0 import (
    EdgeKind,
    OpKind,
    StateAccess,
    StateAccessMode,
    state_access_tensor_view,
)
from .ir1 import CrossGroupRoute, IR1, MemoryInitiator, PairRoute
from .persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateLifetime,
    StateKind,
    canonical_state_staging_value_id,
)
from .state_transfer import (
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
    StateTransferContract,
    StateTransferLike,
)


INTRA_DIE_DAG_SCHEMA_VERSION = "wafer_frontend.intra_die_dag/v1alpha14"
INTRA_DIE_SCHEDULE_SCHEMA_VERSION = "wafer_frontend.intra_die_schedule/v1alpha14"
INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION = "wafer_frontend.intra_die_schedule_set/v1alpha9"
IR2_PROJECTION_RESULT_SCHEMA_VERSION = "wafer_frontend.ir2_projection_result/v1alpha13"
SEMANTIC_FLOW_ID_SCHEMA_VERSION = "wafer_frontend.semantic_flow_identity/v1"
STATE_STAGING_VALUE_SCHEMA_VERSION = "wafer_frontend.state_staging_value/v1alpha1"
STATE_TRANSFER_IR2_ID_SCHEMA_VERSION = (
    "wafer_frontend.state_transfer_ir2_identity/v1"
)


class OriginKind(str, Enum):
    FUSED = "fused"
    ORDINARY = "ordinary"
    STANDALONE_COLLECTIVE = "standalone_collective"
    STATE_IO = "state_io"
    STATE_TRANSFER = "state_transfer"


@dataclass(frozen=True, slots=True)
class FusedNodeOrigin:
    kind: OriginKind
    plan_id: str
    rank: int
    action_id: str

    def validate(self, path: str) -> None:
        if self.kind is not OriginKind.FUSED:
            raise SchemaError("must be fused", path=f"{path}.kind")
        validate_nonempty(self.plan_id, f"{path}.plan_id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.action_id, f"{path}.action_id")


@dataclass(frozen=True, slots=True)
class OrdinaryNodeOrigin:
    kind: OriginKind
    op_id: str
    rank: int

    def validate(self, path: str) -> None:
        if self.kind is not OriginKind.ORDINARY:
            raise SchemaError("must be ordinary", path=f"{path}.kind")
        validate_nonempty(self.op_id, f"{path}.op_id")
        validate_uint64(self.rank, f"{path}.rank")


@dataclass(frozen=True, slots=True)
class StandaloneNodeOrigin:
    kind: OriginKind
    collective_plan_id: str
    rank: int
    action_id: str

    def validate(self, path: str) -> None:
        if self.kind is not OriginKind.STANDALONE_COLLECTIVE:
            raise SchemaError("must be standalone_collective", path=f"{path}.kind")
        validate_nonempty(self.collective_plan_id, f"{path}.collective_plan_id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.action_id, f"{path}.action_id")


@dataclass(frozen=True, slots=True)
class StateIoOrigin:
    """Logical state-access provenance; physical HBM placement stays in IR-1."""

    kind: OriginKind
    state_access_ref: str
    node_ref: str
    rank: int

    def validate(self, path: str) -> None:
        if self.kind is not OriginKind.STATE_IO:
            raise SchemaError("must be state_io", path=f"{path}.kind")
        validate_nonempty(self.state_access_ref, f"{path}.state_access_ref")
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        validate_uint64(self.rank, f"{path}.rank")


@dataclass(frozen=True, slots=True)
class StateTransferOrigin:
    """Typed provenance for transport tasks owned by one transfer contract."""

    kind: OriginKind
    state_transfer_ref: str
    rank: int
    segment_index: int | None = None

    def validate(self, path: str) -> None:
        if self.kind is not OriginKind.STATE_TRANSFER:
            raise SchemaError("must be state_transfer", path=f"{path}.kind")
        validate_nonempty(
            self.state_transfer_ref, f"{path}.state_transfer_ref"
        )
        validate_uint64(self.rank, f"{path}.rank")
        if self.segment_index is not None:
            validate_uint64(
                self.segment_index, f"{path}.segment_index"
            )


NodeOrigin = (
    FusedNodeOrigin
    | OrdinaryNodeOrigin
    | StandaloneNodeOrigin
    | StateIoOrigin
    | StateTransferOrigin
)


def canonical_semantic_flow_id(
    source_send_origin: FusedNodeOrigin | StandaloneNodeOrigin,
    logical_channel: str,
) -> str:
    """Return the one semantic-flow identity shared by every route replica."""

    source_send_origin.validate("source_send_origin")
    validate_nonempty(logical_channel, "logical_channel")
    return stable_artifact_id(
        "semantic_flow",
        {
            "source_send_origin": source_send_origin,
            "logical_channel": logical_channel,
        },
        schema_version=SEMANTIC_FLOW_ID_SCHEMA_VERSION,
    )


def canonical_transit_completion_event(flow_id: str, die_id: int) -> str:
    """Return the local completion event for one flow replica's transit hop."""

    validate_nonempty(flow_id, "flow_id")
    validate_uint64(die_id, "die_id")
    return stable_artifact_id(
        "semantic_transit_completion",
        {"flow_id": flow_id, "die_id": die_id},
        schema_version=SEMANTIC_FLOW_ID_SCHEMA_VERSION,
    )


def canonical_state_transfer_flow_id(
    state_transfer_ref: str,
    segment_index: int | None = None,
) -> str:
    """Return the logical flow identity shared by every transfer route replica."""

    validate_nonempty(state_transfer_ref, "state_transfer_ref")
    if segment_index is not None:
        validate_uint64(segment_index, "segment_index")
        semantic_key = {
            "state_transfer_ref": state_transfer_ref,
            "segment_index": segment_index,
        }
    else:
        semantic_key = {"state_transfer_ref": state_transfer_ref}
    return stable_artifact_id(
        "state_transfer_semantic_flow",
        semantic_key,
        schema_version=STATE_TRANSFER_IR2_ID_SCHEMA_VERSION,
    )


def canonical_state_transfer_payload_id(
    state_transfer_ref: str,
    segment_index: int | None = None,
) -> str:
    """Return the route-global logical payload identity for one state transfer."""

    validate_nonempty(state_transfer_ref, "state_transfer_ref")
    if segment_index is not None:
        validate_uint64(segment_index, "segment_index")
        semantic_key = {
            "state_transfer_ref": state_transfer_ref,
            "segment_index": segment_index,
        }
    else:
        semantic_key = {"state_transfer_ref": state_transfer_ref}
    return stable_artifact_id(
        "state_transfer_payload",
        semantic_key,
        schema_version=STATE_TRANSFER_IR2_ID_SCHEMA_VERSION,
    )


def canonical_state_transfer_task_id(
    state_transfer_ref: str,
    kind: "SemanticTaskKind",
    die_id: int,
    segment_index: int | None = None,
) -> str:
    """Return one route-local state-transfer task identity."""

    validate_nonempty(state_transfer_ref, "state_transfer_ref")
    validate_uint64(die_id, "die_id")
    if segment_index is not None:
        validate_uint64(segment_index, "segment_index")
    if kind not in (
        SemanticTaskKind.SEND,
        SemanticTaskKind.RECV,
        SemanticTaskKind.WAIT,
        SemanticTaskKind.TRANSIT,
    ):
        raise SchemaError(
            "state transfer task identity requires SEND/RECV/WAIT/TRANSIT",
            path="kind",
        )
    segment = "" if segment_index is None else f".segment.{segment_index}"
    return (
        f"task.state_transfer.{state_transfer_ref}{segment}.{kind.value}.die.{die_id}"
    )


def canonical_state_transfer_region_id(
    state_transfer_ref: str,
    die_id: int,
) -> str:
    """Return the one strict transfer region identity on a selected route die."""

    validate_nonempty(state_transfer_ref, "state_transfer_ref")
    validate_uint64(die_id, "die_id")
    return f"region.state_transfer.{state_transfer_ref}.die.{die_id}"


def canonical_state_transfer_completion_event(
    state_transfer_ref: str,
    kind: "SemanticTaskKind",
    die_id: int,
    segment_index: int | None = None,
) -> str:
    """Return one route-local transfer completion event identity."""

    task_id = canonical_state_transfer_task_id(
        state_transfer_ref, kind, die_id, segment_index
    )
    return stable_artifact_id(
        "state_transfer_completion",
        {"task_id": task_id},
        schema_version=STATE_TRANSFER_IR2_ID_SCHEMA_VERSION,
    )


class SemanticTaskKind(str, Enum):
    DMA_IN = "dma_in"
    COMP = "comp"
    LOCAL_COPY = "local_copy"
    SEND = "send"
    RECV = "recv"
    REDUCE = "reduce"
    WAIT = "wait"
    BARRIER = "barrier"
    TRANSIT = "transit"
    DMA_OUT = "dma_out"


class RegionLowering(str, Enum):
    JSON_COARSE = "json_coarse"
    ISA_REGION = "isa_region"
    STRICT_ACTIONS = "strict_actions"
    STRICT_STATE_IO = "strict_state_io"
    STRICT_STATE_TRANSFER = "strict_state_transfer"


def canonical_state_task_id(
    state_access_ref: str,
    kind: SemanticTaskKind,
) -> str:
    """Return the semantic task identity for one persistent-state transfer."""

    validate_nonempty(state_access_ref, "state_access_ref")
    if kind not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT):
        raise SchemaError(
            "state task identity requires DMA_IN or DMA_OUT",
            path="kind",
        )
    return f"task.state.{state_access_ref}.{kind.value}"


def canonical_state_region_id(
    state_access_ref: str,
    kind: SemanticTaskKind,
) -> str:
    """Return the strict lowering-region identity for one state DMA task."""

    validate_nonempty(state_access_ref, "state_access_ref")
    if kind not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT):
        raise SchemaError(
            "state region identity requires DMA_IN or DMA_OUT",
            path="kind",
        )
    return f"region.state.{state_access_ref}.{kind.value}"


def canonical_state_access_view(
    ir1: IR1,
    access: StateAccess,
    declaration: PersistentStateDecl,
    dma_kind: SemanticTaskKind,
    path: str = "state_access_view",
) -> tuple[
    tuple[int, ...],
    str,
    tuple[int, ...],
    tuple[int, ...],
]:
    """Derive the logical staging domain and rank-global DMA shard view."""

    if dma_kind is SemanticTaskKind.DMA_IN:
        direction = "read"
    elif dma_kind is SemanticTaskKind.DMA_OUT:
        direction = "write"
    else:
        raise SchemaError(
            "state access view requires DMA_IN or DMA_OUT",
            path=f"{path}.dma_kind",
        )
    tensor_ref = declaration.identity.tensor_ref
    if tensor_ref is None:
        dma_offset, dma_shape = state_access_tensor_view(
            access,
            declaration,
            direction,
            path=path,
        )
        return (
            declaration.shape,
            declaration.layout,
            dma_offset,
            dma_shape,
        )

    if any(
        value is not None
        for value in (
            access.read_offset,
            access.read_shape,
            access.write_offset,
            access.write_shape,
        )
    ):
        raise SchemaError(
            "parameter accesses must use their canonical whole shard view",
            path=path,
        )

    value_index = {value.id: value for value in ir1.values}
    node_index = {node.id: node for node in ir1.nodes}
    group_index = {group.id: group for group in ir1.groups}
    source = value_index.get(tensor_ref)
    node = node_index.get(access.node_ref)
    if source is None or node is None:
        raise SchemaError(
            "parameter state view references a missing IR-1 tensor or node",
            path=path,
        )
    group = group_index.get(node.execution_group_ref)
    if group is None:
        raise SchemaError(
            "parameter state view references a missing execution group",
            path=path,
        )
    placement = next(
        (
            candidate
            for candidate in group.placements
            if candidate.rank == access.rank
        ),
        None,
    )
    if placement is None:
        raise SchemaError(
            "parameter state rank is absent from its execution group",
            path=path,
        )
    if (
        source.dtype is not declaration.dtype
        or source.logical_layout != declaration.layout
        or source.sharding.mesh_ref != group.mesh_ref
        or len(source.shape) != len(declaration.shape)
        or source.sharding.partial
        or any(
            axis not in (None, group.axis)
            for axis in source.sharding.dim_map
        )
    ):
        raise SchemaError(
            "parameter declaration disagrees with its IR-1 tensor domain",
            path=path,
        )

    mapped_dimensions = tuple(
        index
        for index, axis in enumerate(source.sharding.dim_map)
        if axis is group.axis
    )
    if len(mapped_dimensions) > 1:
        raise SchemaError(
            "parameter tensor may shard the execution-group axis once",
            path=path,
        )
    offset = [0] * len(source.shape)
    expected_local_shape = list(source.shape)
    if mapped_dimensions:
        if (
            len(group.logical_shape) != 1
            or len(placement.logical_coord) != 1
        ):
            raise SchemaError(
                "parameter shard view requires a one-dimensional group",
                path=path,
            )
        dimension = mapped_dimensions[0]
        group_size = group.logical_shape[0]
        if source.shape[dimension] % group_size != 0:
            raise SchemaError(
                "parameter tensor dimension is not divisible by group size",
                path=path,
            )
        expected_local_shape[dimension] //= group_size
        offset[dimension] = (
            placement.logical_coord[0] * expected_local_shape[dimension]
        )
    if tuple(expected_local_shape) != declaration.shape:
        raise SchemaError(
            "parameter declaration shape is not the canonical rank shard",
            path=path,
        )
    dma_offset = tuple(offset)
    if any(
        dma_offset[index] + declaration.shape[index]
        > source.shape[index]
        for index in range(len(source.shape))
    ):
        raise SchemaError(
            "parameter DMA shard view lies outside its IR-1 tensor domain",
            path=path,
        )
    return (
        source.shape,
        source.logical_layout,
        dma_offset,
        declaration.shape,
    )


@dataclass(frozen=True, slots=True)
class TensorSlice:
    value_id: str
    offset: tuple[int, ...]
    shape: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_id, f"{path}.value_id")
        if not self.shape or len(self.offset) != len(self.shape):
            raise SchemaError("offset and shape must have equal non-zero rank", path=path)
        for field_name in ("offset", "shape"):
            for index, value in enumerate(getattr(self, field_name)):
                validate_uint64(value, f"{path}.{field_name}[{index}]")
                if field_name == "shape" and value == 0:
                    raise SchemaError("must be greater than zero", path=f"{path}.shape[{index}]")


def dense_row_major_view_byte_addend(
    root: TensorSlice,
    view: TensorSlice,
    dtype: DType,
    *,
    path: str = "tensor_slice",
) -> int:
    """Return the byte addend of one contiguous view inside a tight root.

    Schedule-v1alpha8 freezes every BufferBinding as one tight dense row-major
    backing. A task use is a logical view of that backing, not another
    allocation. Backends with one address/length operand may consume only a
    view whose rectangular logical elements form one contiguous linear span.
    """

    if type(root) is not TensorSlice or type(view) is not TensorSlice:
        raise SchemaError("root and view must be TensorSlice values", path=path)
    root.validate(f"{path}.root")
    view.validate(f"{path}.view")
    if root.value_id != view.value_id:
        raise SchemaError(
            "view and root must reference the same value",
            path=f"{path}.view.value_id",
        )
    if len(root.shape) != len(view.shape):
        raise SchemaError(
            "view and root must have equal rank",
            path=f"{path}.view",
        )
    element_bytes = {
        DType.FP16: 2,
        DType.FP32: 4,
        DType.INT32: 4,
    }.get(dtype)
    if element_bytes is None:
        raise SchemaError("unsupported dense-view dtype", path=f"{path}.dtype")

    relative: list[int] = []
    for root_offset, root_extent, view_offset, view_extent in zip(
        root.offset, root.shape, view.offset, view.shape
    ):
        if (
            root_offset > UINT64_MAX - root_extent
            or view_offset > UINT64_MAX - view_extent
        ):
            raise SchemaError("dense view bounds overflow uint64", path=path)
        root_end = root_offset + root_extent
        view_end = view_offset + view_extent
        if view_offset < root_offset or view_end > root_end:
            raise SchemaError(
                "view must be contained in its root backing",
                path=f"{path}.view",
            )
        relative.append(view_offset - root_offset)

    strides = [1] * len(root.shape)
    for axis in range(len(root.shape) - 2, -1, -1):
        if strides[axis + 1] > UINT64_MAX // root.shape[axis + 1]:
            raise SchemaError(
                "dense root stride overflows uint64",
                path=f"{path}.root.shape",
            )
        strides[axis] = strides[axis + 1] * root.shape[axis + 1]

    first = 0
    last = 0
    view_elements = 1
    for axis, stride in enumerate(strides):
        if relative[axis] > UINT64_MAX // stride:
            raise SchemaError("dense view offset overflows uint64", path=path)
        first_term = relative[axis] * stride
        last_coordinate = relative[axis] + view.shape[axis] - 1
        if last_coordinate > UINT64_MAX // stride:
            raise SchemaError("dense view extent overflows uint64", path=path)
        last_term = last_coordinate * stride
        if first > UINT64_MAX - first_term or last > UINT64_MAX - last_term:
            raise SchemaError("dense view span overflows uint64", path=path)
        first += first_term
        last += last_term
        if view_elements > UINT64_MAX // view.shape[axis]:
            raise SchemaError("dense view element count overflows uint64", path=path)
        view_elements *= view.shape[axis]
    if last - first + 1 != view_elements:
        raise SchemaError(
            "view is not contiguous in its dense row-major root backing",
            path=f"{path}.view",
        )
    if first > UINT64_MAX // element_bytes:
        raise SchemaError("dense view byte addend overflows uint64", path=path)
    if view_elements > UINT64_MAX // element_bytes:
        raise SchemaError("dense view byte length overflows uint64", path=path)
    return first * element_bytes


@dataclass(frozen=True, slots=True)
class IntraDieValue:
    id: str
    origin_value_id: str
    shape: tuple[int, ...]
    dtype: DType
    logical_layout: str
    sharding: Sharding
    alias_set: str | None
    producer_tasks: tuple[str, ...]
    consumer_tasks: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.origin_value_id, f"{path}.origin_value_id")
        if not self.shape:
            raise SchemaError("must have non-zero rank", path=f"{path}.shape")
        for index, dimension in enumerate(self.shape):
            validate_uint64(dimension, f"{path}.shape[{index}]")
            if dimension == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.shape[{index}]")
        validate_nonempty(self.logical_layout, f"{path}.logical_layout")
        self.sharding.validate(f"{path}.sharding")
        if self.alias_set is not None:
            validate_nonempty(self.alias_set, f"{path}.alias_set")
        if len(set(self.producer_tasks)) != len(self.producer_tasks):
            raise SchemaError("contains duplicate task ids", path=f"{path}.producer_tasks")
        for index, producer_task in enumerate(self.producer_tasks):
            validate_nonempty(producer_task, f"{path}.producer_tasks[{index}]")
        if len(set(self.consumer_tasks)) != len(self.consumer_tasks):
            raise SchemaError("contains duplicate task ids", path=f"{path}.consumer_tasks")


@dataclass(frozen=True, slots=True)
class DmaContract:
    """Bounded state transfer with exactly one local SRAM endpoint."""

    state_ref: str
    local_value_ref: str
    state_offset_bytes: int
    access_task_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        validate_nonempty(self.local_value_ref, f"{path}.local_value_ref")
        validate_uint64(self.state_offset_bytes, f"{path}.state_offset_bytes")
        if not self.access_task_refs:
            raise SchemaError(
                "must identify at least one state-access task",
                path=f"{path}.access_task_refs",
            )
        if len(set(self.access_task_refs)) != len(self.access_task_refs):
            raise SchemaError(
                "contains duplicate task ids", path=f"{path}.access_task_refs"
            )
        for index, task_ref in enumerate(self.access_task_refs):
            validate_nonempty(task_ref, f"{path}.access_task_refs[{index}]")


@dataclass(frozen=True, slots=True)
class StateStagingValue:
    """SRAM-visible value for persistent state without an ordinary tensor id."""

    id: str
    state_access_ref: str
    state_ref: str
    shape: tuple[int, ...]
    dtype: DType
    logical_layout: str
    producer_tasks: tuple[str, ...]
    consumer_tasks: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        state_access_ref: str,
        state_ref: str,
        shape: tuple[int, ...],
        dtype: DType,
        logical_layout: str,
        producer_tasks: tuple[str, ...],
        consumer_tasks: tuple[str, ...],
    ) -> "StateStagingValue":
        semantic_key = {
            "state_access_ref": state_access_ref,
            "state_ref": state_ref,
            "shape": shape,
            "dtype": dtype,
            "logical_layout": logical_layout,
        }
        result = cls(
            id=canonical_state_staging_value_id(state_access_ref),
            producer_tasks=producer_tasks,
            consumer_tasks=consumer_tasks,
            **semantic_key,
        )
        result.validate("state_staging_value")
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "state_access_ref": self.state_access_ref,
            "state_ref": self.state_ref,
            "shape": self.shape,
            "dtype": self.dtype,
            "logical_layout": self.logical_layout,
        }

    def validate(self, path: str) -> None:
        validate_nonempty(self.state_access_ref, f"{path}.state_access_ref")
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        if not self.shape:
            raise SchemaError("must have non-zero rank", path=f"{path}.shape")
        for index, dimension in enumerate(self.shape):
            validate_uint64(dimension, f"{path}.shape[{index}]")
            if dimension == 0:
                raise SchemaError(
                    "must be greater than zero", path=f"{path}.shape[{index}]"
                )
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        validate_nonempty(self.logical_layout, f"{path}.logical_layout")
        for field_name in ("producer_tasks", "consumer_tasks"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError(
                    "contains duplicate task ids", path=f"{path}.{field_name}"
                )
            for index, task_ref in enumerate(refs):
                validate_nonempty(task_ref, f"{path}.{field_name}[{index}]")
        expected = canonical_state_staging_value_id(self.state_access_ref)
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class SemanticTask:
    id: str
    kind: SemanticTaskKind
    origin_ref: NodeOrigin
    region_id: str | None
    op_kind: OpKind | None
    member_id: str | None
    flow_id: str | None
    chunk_id: int | None
    collective_step: int | None
    source_rank: int | None
    destination_rank: int | None
    tensor_slice: TensorSlice | None
    bytes: int
    dtype: DType | None
    shape: tuple[int, ...]
    read_values: tuple[str, ...]
    write_values: tuple[str, ...]
    compute: ComputeContract | None
    reduction: ReductionContract | None
    sync: SyncContract | None
    deps: tuple[str, ...]
    dma: DmaContract | None = None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        self.origin_ref.validate(f"{path}.origin_ref")
        is_dma = self.kind in (
            SemanticTaskKind.DMA_IN,
            SemanticTaskKind.DMA_OUT,
        )
        if is_dma:
            if type(self.origin_ref) is not StateIoOrigin:
                raise SchemaError(
                    "state DMA requires StateIoOrigin", path=f"{path}.origin_ref"
                )
            if type(self.dma) is not DmaContract:
                raise SchemaError(
                    "state DMA requires DmaContract", path=f"{path}.dma"
                )
            self.dma.validate(f"{path}.dma")
            if any(
                value is not None
                for value in (
                    self.op_kind,
                    self.member_id,
                    self.flow_id,
                    self.chunk_id,
                    self.collective_step,
                    self.source_rank,
                    self.destination_rank,
                    self.compute,
                    self.reduction,
                    self.sync,
                )
            ):
                raise SchemaError(
                    "state DMA cannot carry compute/flow/collective/sync fields",
                    path=path,
                )
            if (
                self.tensor_slice is None
                or self.dtype is None
                or not self.shape
                or self.bytes == 0
            ):
                raise SchemaError(
                    "state DMA requires a non-empty local payload",
                    path=path,
                )
            if (
                self.tensor_slice.value_id != self.dma.local_value_ref
                or self.tensor_slice.shape != self.shape
            ):
                raise SchemaError(
                    "DMA local value and shape must equal tensor_slice",
                    path=f"{path}.tensor_slice",
                )
            element_bytes = {DType.FP16: 2, DType.FP32: 4}.get(self.dtype)
            if element_bytes is None or math.prod(self.shape) * element_bytes != self.bytes:
                raise SchemaError(
                    "DMA bytes must equal the tight tensor payload",
                    path=f"{path}.bytes",
                )
            expected_values = {
                SemanticTaskKind.DMA_IN: (
                    (),
                    (self.dma.local_value_ref,),
                ),
                SemanticTaskKind.DMA_OUT: (
                    (self.dma.local_value_ref,),
                    (),
                ),
            }[self.kind]
            if (self.read_values, self.write_values) != expected_values:
                raise SchemaError(
                    "DMA_IN must have one local WRITE endpoint and DMA_OUT one local READ endpoint",
                    path=path,
                )
        else:
            if isinstance(self.origin_ref, StateIoOrigin):
                raise SchemaError(
                    "StateIoOrigin is valid only for DMA_IN/DMA_OUT",
                    path=f"{path}.origin_ref",
                )
            if self.dma is not None:
                raise SchemaError(
                    "DmaContract is valid only for DMA_IN/DMA_OUT",
                    path=f"{path}.dma",
                )
        if isinstance(self.origin_ref, StateTransferOrigin):
            if self.kind not in (
                SemanticTaskKind.SEND,
                SemanticTaskKind.RECV,
                SemanticTaskKind.WAIT,
                SemanticTaskKind.TRANSIT,
            ):
                raise SchemaError(
                    "StateTransferOrigin requires SEND/RECV/WAIT/TRANSIT",
                    path=f"{path}.kind",
                )
            if (
                self.op_kind is not OpKind.P2P
                or self.member_id is not None
                or self.chunk_id is not None
                or self.collective_step is not None
                or self.compute is not None
                or self.reduction is not None
                or self.dma is not None
                or self.sync is None
            ):
                raise SchemaError(
                    "state transfer carries only P2P payload/sync fields",
                    path=path,
                )
            expected_values = {
                SemanticTaskKind.SEND: (1, 0),
                SemanticTaskKind.RECV: (0, 1),
                SemanticTaskKind.WAIT: (0, 0),
                SemanticTaskKind.TRANSIT: (0, 0),
            }[self.kind]
            if (len(self.read_values), len(self.write_values)) != expected_values:
                raise SchemaError(
                    "state transfer endpoint direction is not exact",
                    path=path,
                )
            if self.kind is SemanticTaskKind.WAIT and any(
                value is not None
                for value in (
                    self.flow_id,
                    self.source_rank,
                    self.destination_rank,
                )
            ):
                raise SchemaError(
                    "state transfer WAIT cannot carry flow endpoints",
                    path=path,
                )
        for field_name in ("region_id", "member_id", "flow_id"):
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty(value, f"{path}.{field_name}")
        for field_name in ("chunk_id", "collective_step", "source_rank", "destination_rank"):
            value = getattr(self, field_name)
            if value is not None:
                validate_uint64(value, f"{path}.{field_name}")
        validate_uint64(self.bytes, f"{path}.bytes")
        for index, dimension in enumerate(self.shape):
            validate_uint64(dimension, f"{path}.shape[{index}]")
            if dimension == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.shape[{index}]")
        for field_name in ("read_values", "write_values"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")
        if self.tensor_slice is not None:
            self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.reduction is not None:
            self.reduction.validate(f"{path}.reduction")
        if self.compute is not None:
            self.compute.validate(f"{path}.compute")
            if tuple(item.value_id for item in self.compute.inputs) != self.read_values:
                raise SchemaError("inputs must exactly match read_values", path=f"{path}.compute.inputs")
            if tuple(item.value_id for item in self.compute.outputs) != self.write_values:
                raise SchemaError("outputs must exactly match write_values", path=f"{path}.compute.outputs")
        planned_origin = isinstance(
            self.origin_ref, (FusedNodeOrigin, StandaloneNodeOrigin)
        )
        if planned_origin and self.sync is None:
            raise SchemaError("planned task requires sync identity", path=f"{path}.sync")
        if self.sync is not None:
            if self.kind is SemanticTaskKind.TRANSIT:
                validate_nonempty(
                    self.sync.completion_event, f"{path}.sync.completion_event"
                )
                if self.sync.wait_event is not None or self.sync.barrier is not None:
                    raise SchemaError("TRANSIT cannot carry wait/barrier binding", path=f"{path}.sync")
            elif self.kind.value in {kind.value for kind in FusionActionKind}:
                self.sync.validate_for_kind(
                    FusionActionKind(self.kind.value), f"{path}.sync"
                )
        if self.kind in (SemanticTaskKind.SEND, SemanticTaskKind.RECV, SemanticTaskKind.TRANSIT):
            if self.flow_id is None or self.tensor_slice is None or self.dtype is None or self.bytes == 0:
                raise SchemaError("transport task requires flow, slice, dtype and non-zero bytes", path=path)
            if self.source_rank is None or self.destination_rank is None:
                raise SchemaError("transport task requires source and destination ranks", path=path)
        elif self.kind in (SemanticTaskKind.WAIT, SemanticTaskKind.BARRIER):
            if self.bytes != 0 or self.dtype is not None or self.tensor_slice is not None:
                raise SchemaError("WAIT/BARRIER cannot carry a payload", path=path)
        elif self.bytes > 0 and self.dtype is None:
            raise SchemaError("payload task requires dtype", path=f"{path}.dtype")
        if self.kind is SemanticTaskKind.REDUCE:
            if self.reduction is None:
                raise SchemaError("is required for REDUCE", path=f"{path}.reduction")
            if self.dtype is not self.reduction.input_dtype:
                raise SchemaError("dtype must equal reduction input_dtype", path=f"{path}.dtype")
        elif self.reduction is not None:
            raise SchemaError("is only valid for REDUCE", path=f"{path}.reduction")
        if self.kind is SemanticTaskKind.COMP:
            if self.compute is None:
                raise SchemaError("COMP requires compute contract", path=f"{path}.compute")
        elif self.compute is not None and self.kind is not SemanticTaskKind.COMP:
            raise SchemaError("is only valid for COMP", path=f"{path}.compute")


@dataclass(frozen=True, slots=True)
class SemanticFlow:
    id: str
    logical_channel: str
    pair_route_ref: str
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    die_path: tuple[int, ...]
    tensor_slice: TensorSlice
    bytes: int
    dtype: DType
    task_ids: tuple[str, ...]

    def validate(
        self,
        path: str,
        *,
        allow_equal_ranks_for_state_transfer: bool | None = None,
    ) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.logical_channel, f"{path}.logical_channel")
        validate_nonempty(self.pair_route_ref, f"{path}.pair_route_ref")
        if allow_equal_ranks_for_state_transfer is None:
            allow_equal_ranks_for_state_transfer = (
                self.logical_channel.startswith("state_transfer.")
            )
        for field_name in ("source_rank", "destination_rank", "source_die", "destination_die"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if (
            self.source_rank == self.destination_rank
            and not allow_equal_ranks_for_state_transfer
        ) or self.source_die == self.destination_die:
            raise SchemaError("flow endpoints must differ", path=path)
        if (
            len(self.die_path) < 2
            or self.die_path[0] != self.source_die
            or self.die_path[-1] != self.destination_die
        ):
            raise SchemaError(
                "die_path must preserve source, intermediate and destination dies",
                path=f"{path}.die_path",
            )
        if len(set(self.die_path)) != len(self.die_path):
            raise SchemaError("die_path cannot contain a cycle", path=f"{path}.die_path")
        for index, die_id in enumerate(self.die_path):
            validate_uint64(die_id, f"{path}.die_path[{index}]")
        self.tensor_slice.validate(f"{path}.tensor_slice")
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise SchemaError("must contain unique local task ids", path=f"{path}.task_ids")


@dataclass(frozen=True, slots=True)
class IntraDieRegion:
    id: str
    fusion_plan_id: str | None
    standalone_collective_plan_id: str | None
    lowering: RegionLowering
    task_ids: tuple[str, ...]
    state_transfer_ref: str | None = None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        for field_name in (
            "fusion_plan_id",
            "standalone_collective_plan_id",
            "state_transfer_ref",
        ):
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty(value, f"{path}.{field_name}")
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise SchemaError("must contain unique task ids", path=f"{path}.task_ids")
        if self.lowering is RegionLowering.ISA_REGION:
            if (
                self.fusion_plan_id is None
                or self.standalone_collective_plan_id is not None
                or self.state_transfer_ref is not None
            ):
                raise SchemaError("ISA region requires only fusion_plan_id", path=path)
        elif self.lowering is RegionLowering.STRICT_ACTIONS:
            if (
                self.standalone_collective_plan_id is None
                or self.fusion_plan_id is not None
                or self.state_transfer_ref is not None
            ):
                raise SchemaError("strict-actions region requires only standalone plan", path=path)
        elif self.lowering is RegionLowering.STRICT_STATE_IO:
            if (
                self.fusion_plan_id is not None
                or self.standalone_collective_plan_id is not None
                or self.state_transfer_ref is not None
            ):
                raise SchemaError("strict-state-io region cannot reference a plan", path=path)
        elif self.lowering is RegionLowering.STRICT_STATE_TRANSFER:
            if (
                self.state_transfer_ref is None
                or self.fusion_plan_id is not None
                or self.standalone_collective_plan_id is not None
            ):
                raise SchemaError(
                    "strict-state-transfer region requires only state_transfer_ref",
                    path=path,
                )
        elif (
            self.fusion_plan_id is not None
            or self.standalone_collective_plan_id is not None
            or self.state_transfer_ref is not None
        ):
            raise SchemaError("coarse region cannot reference a collective plan", path=path)


def _slices_overlap(left: TensorSlice, right: TensorSlice) -> bool:
    return all(
        left.offset[axis] < right.offset[axis] + right.shape[axis]
        and right.offset[axis] < left.offset[axis] + left.shape[axis]
        for axis in range(len(left.shape))
    )


def _validate_rectangular_slices(
    slices: tuple[TensorSlice, ...],
    *,
    value_id: str,
    value_shape: tuple[int, ...],
    require_full_cover: bool,
    path: str,
) -> None:
    for index, tensor_slice in enumerate(slices):
        if tensor_slice.value_id != value_id:
            raise SchemaError(
                "writer slice must name the written value",
                path=f"{path}[{index}].value_id",
            )
        if len(tensor_slice.shape) != len(value_shape) or any(
            tensor_slice.offset[axis] + tensor_slice.shape[axis] > value_shape[axis]
            for axis in range(len(tensor_slice.shape))
        ):
            raise SchemaError("writer slice lies outside its value", path=f"{path}[{index}]")
        for previous_index, previous in enumerate(slices[:index]):
            if _slices_overlap(previous, tensor_slice):
                raise SchemaError(
                    f"writer slices overlap with entry {previous_index}",
                    path=f"{path}[{index}]",
                )
    if require_full_cover and sum(math.prod(item.shape) for item in slices) != math.prod(
        value_shape
    ):
        raise SchemaError("writer slices do not fully cover the value", path=path)


def _producer_sort_key(task: SemanticTask) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    tensor_slice = task.tensor_slice
    return (
        tensor_slice.offset if tensor_slice is not None else (),
        tensor_slice.shape if tensor_slice is not None else (),
        task.id,
    )


@dataclass(frozen=True, slots=True)
class IntraDieDAG:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    die_id: int
    fusion_plan_ids: tuple[str, ...]
    standalone_collective_plan_ids: tuple[str, ...]
    ordinary_node_ids: tuple[str, ...]
    tasks: tuple[SemanticTask, ...]
    values: tuple[IntraDieValue, ...]
    flows: tuple[SemanticFlow, ...]
    regions: tuple[IntraDieRegion, ...]
    source_state_manifest_id: str | None = None
    state_access_ids: tuple[str, ...] = ()
    state_staging_values: tuple[StateStagingValue, ...] = ()
    state_transfer_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "IntraDieDAG":
        semantic_key.setdefault("source_state_manifest_id", None)
        semantic_key.setdefault("state_access_ids", ())
        semantic_key.setdefault("state_staging_values", ())
        semantic_key.setdefault("state_transfer_ids", ())
        return cls(
            schema_version=INTRA_DIE_DAG_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("intra_die_dag", semantic_key, schema_version=INTRA_DIE_DAG_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "die_id", "fusion_plan_ids", "standalone_collective_plan_ids",
            "ordinary_node_ids", "tasks", "values", "flows", "regions",
            "source_state_manifest_id", "state_access_ids", "state_staging_values",
            "state_transfer_ids",
        )}

    def validate(self, path: str = "intra_die_dag") -> None:
        if self.schema_version != INTRA_DIE_DAG_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        validate_uint64(self.die_id, f"{path}.die_id")
        for field_name in (
            "fusion_plan_ids",
            "standalone_collective_plan_ids",
            "ordinary_node_ids",
            "state_transfer_ids",
        ):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")
        if self.source_state_manifest_id is not None:
            validate_nonempty(
                self.source_state_manifest_id,
                f"{path}.source_state_manifest_id",
            )
        if len(set(self.state_access_ids)) != len(self.state_access_ids):
            raise SchemaError(
                "contains duplicate state access ids",
                path=f"{path}.state_access_ids",
            )
        for index, access_id in enumerate(self.state_access_ids):
            validate_nonempty(access_id, f"{path}.state_access_ids[{index}]")
        if self.state_staging_values != tuple(
            sorted(
                self.state_staging_values,
                key=lambda value: (value.state_access_ref, value.id),
            )
        ):
            raise SchemaError(
                "must use canonical access/id order",
                path=f"{path}.state_staging_values",
            )
        task_index = validate_dependency_dag(self.tasks, f"{path}.tasks")
        state_tasks = tuple(
            task
            for task in self.tasks
            if isinstance(task.origin_ref, StateIoOrigin)
        )
        if (
            self.state_access_ids
            or self.state_staging_values
            or state_tasks
        ) and self.source_state_manifest_id is None:
            raise SchemaError(
                "state projection requires persistent-state manifest provenance",
                path=f"{path}.source_state_manifest_id",
            )
        expected_state_access_ids = tuple(
            dict.fromkeys(
                task.origin_ref.state_access_ref
                for task in state_tasks
            )
        )
        if set(self.state_access_ids) != set(expected_state_access_ids):
            raise SchemaError(
                "state_access_ids must exactly match local StateIoOrigin accesses",
                path=f"{path}.state_access_ids",
            )
        value_index = validate_unique_ids(self.values, f"{path}.values")
        staging_index = validate_unique_ids(
            self.state_staging_values, f"{path}.state_staging_values"
        )
        if set(value_index).intersection(staging_index):
            raise SchemaError("ordinary and state values must have disjoint ids", path=path)
        local_value_index = dict(value_index)
        local_value_index.update(staging_index)
        flow_index = validate_unique_ids(self.flows, f"{path}.flows")
        if len({flow.logical_channel for flow in self.flows}) != len(self.flows):
            raise SchemaError(
                "logical channels must be unique within a die DAG",
                path=f"{path}.flows",
            )
        region_index = validate_unique_ids(self.regions, f"{path}.regions")
        for index, task in enumerate(self.tasks):
            task.validate(f"{path}.tasks[{index}]")
            if task.region_id is None:
                raise SchemaError(
                    "every semantic task must belong to exactly one lowering region",
                    path=f"{path}.tasks[{index}].region_id",
                )
            if task.region_id not in region_index:
                raise SchemaError("dangling region", path=f"{path}.tasks[{index}].region_id")
            if task.flow_id is not None and task.flow_id not in flow_index:
                raise SchemaError("dangling flow", path=f"{path}.tasks[{index}].flow_id")
            if not set(task.read_values + task.write_values).issubset(local_value_index):
                raise SchemaError("contains a dangling value", path=f"{path}.tasks[{index}]")
            origin = task.origin_ref
            if isinstance(origin, StateIoOrigin):
                expected_task_id = canonical_state_task_id(origin.state_access_ref, task.kind)
                expected_region_id = canonical_state_region_id(origin.state_access_ref, task.kind)
                if task.id != expected_task_id or task.region_id != expected_region_id:
                    raise SchemaError(
                        "state DMA task/region identity is not canonical",
                        path=f"{path}.tasks[{index}]",
                    )
            if isinstance(origin, FusedNodeOrigin) and origin.plan_id not in self.fusion_plan_ids:
                raise SchemaError("dangling fusion plan origin", path=f"{path}.tasks[{index}].origin_ref.plan_id")
            if isinstance(origin, StandaloneNodeOrigin) and origin.collective_plan_id not in self.standalone_collective_plan_ids:
                raise SchemaError("dangling standalone plan origin", path=f"{path}.tasks[{index}].origin_ref.collective_plan_id")
            if isinstance(origin, OrdinaryNodeOrigin) and origin.op_id not in self.ordinary_node_ids:
                raise SchemaError("dangling ordinary op origin", path=f"{path}.tasks[{index}].origin_ref.op_id")
            if isinstance(origin, StateTransferOrigin):
                if origin.state_transfer_ref not in self.state_transfer_ids:
                    raise SchemaError(
                        "dangling state transfer origin",
                        path=(
                            f"{path}.tasks[{index}]"
                            ".origin_ref.state_transfer_ref"
                        ),
                    )
                if (
                    task.id
                    != canonical_state_transfer_task_id(
                        origin.state_transfer_ref, task.kind, self.die_id,
                        origin.segment_index,
                    )
                    or task.region_id
                    != canonical_state_transfer_region_id(
                        origin.state_transfer_ref, self.die_id
                    )
                ):
                    raise SchemaError(
                        "state transfer task/region identity is not canonical",
                        path=f"{path}.tasks[{index}]",
                    )
        actual_state_transfer_ids = {
            task.origin_ref.state_transfer_ref
            for task in self.tasks
            if isinstance(task.origin_ref, StateTransferOrigin)
        }
        if actual_state_transfer_ids != set(self.state_transfer_ids):
            raise SchemaError(
                "state_transfer_ids must exactly match local transfer origins",
                path=f"{path}.state_transfer_ids",
            )
        for index, value in enumerate(self.values):
            value.validate(f"{path}.values[{index}]")
            actual_writers = tuple(
                sorted(
                    (task for task in self.tasks if value.id in task.write_values),
                    key=_producer_sort_key,
                )
            )
            actual_producer_ids = tuple(task.id for task in actual_writers)
            if value.producer_tasks != actual_producer_ids:
                raise SchemaError(
                    "producer_tasks must exactly and canonically name every writer",
                    path=f"{path}.values[{index}].producer_tasks",
                )
            if len(actual_writers) > 1:
                origins = tuple(task.origin_ref for task in actual_writers)
                if all(isinstance(origin, StandaloneNodeOrigin) for origin in origins):
                    if len(
                        {origin.collective_plan_id for origin in origins}
                    ) != 1:
                        raise SchemaError(
                            "multiple writers must belong to one standalone plan",
                            path=f"{path}.values[{index}].producer_tasks",
                        )
                    require_full_cover = True
                elif all(isinstance(origin, FusedNodeOrigin) for origin in origins):
                    if len({origin.plan_id for origin in origins}) != 1:
                        raise SchemaError(
                            "multiple writers must belong to one fusion plan",
                            path=f"{path}.values[{index}].producer_tasks",
                        )
                    # A fused RS die may physically own only a subset of global chunks.
                    require_full_cover = False
                else:
                    raise SchemaError(
                        "ordinary or mixed-origin values cannot have multiple writers",
                        path=f"{path}.values[{index}].producer_tasks",
                    )
                writer_slices = tuple(
                    task.tensor_slice for task in actual_writers if task.tensor_slice is not None
                )
                if len(writer_slices) != len(actual_writers):
                    raise SchemaError(
                        "multiple writers require explicit tensor slices",
                        path=f"{path}.values[{index}].producer_tasks",
                    )
                _validate_rectangular_slices(
                    writer_slices,
                    value_id=value.id,
                    value_shape=value.shape,
                    require_full_cover=require_full_cover,
                    path=f"{path}.values[{index}].producer_tasks",
                )
            for consumer_id in value.consumer_tasks:
                consumer = task_index.get(consumer_id)
                if consumer is None or value.id not in consumer.read_values:
                    raise SchemaError("value consumer is dangling or does not read value", path=f"{path}.values[{index}].consumer_tasks")
        for index, value in enumerate(self.state_staging_values):
            value.validate(f"{path}.state_staging_values[{index}]")
            if value.state_access_ref not in self.state_access_ids:
                raise SchemaError(
                    "staging value references an undeclared state access",
                    path=f"{path}.state_staging_values[{index}].state_access_ref",
                )
            actual_writers = tuple(
                sorted(
                    (
                        task
                        for task in self.tasks
                        if value.id in task.write_values
                    ),
                    key=_producer_sort_key,
                )
            )
            if value.producer_tasks != tuple(task.id for task in actual_writers):
                raise SchemaError(
                    "producer_tasks must exactly and canonically name every writer",
                    path=f"{path}.state_staging_values[{index}].producer_tasks",
                )
            if len(actual_writers) > 1:
                writer_path = (
                    f"{path}.state_staging_values[{index}].producer_tasks"
                )
                if any(
                    writer.kind is not SemanticTaskKind.RECV
                    or not isinstance(
                        writer.origin_ref, StateTransferOrigin
                    )
                    or writer.tensor_slice is None
                    for writer in actual_writers
                ) or len(
                    {
                        (
                            writer.origin_ref.state_transfer_ref,
                            writer.origin_ref.segment_index,
                        )
                        for writer in actual_writers
                        if isinstance(
                            writer.origin_ref, StateTransferOrigin
                        )
                    }
                ) != len(actual_writers):
                    raise SchemaError(
                        "multiple state writers require distinct transfer/segment RECV tasks",
                        path=writer_path,
                    )
                _validate_rectangular_slices(
                    tuple(
                        writer.tensor_slice
                        for writer in actual_writers
                        if writer.tensor_slice is not None
                    ),
                    value_id=value.id,
                    value_shape=value.shape,
                    require_full_cover=False,
                    path=writer_path,
                )
            for consumer_id in value.consumer_tasks:
                consumer = task_index.get(consumer_id)
                if consumer is None or value.id not in consumer.read_values:
                    raise SchemaError(
                        "state consumer is dangling or does not read the staging value",
                        path=f"{path}.state_staging_values[{index}].consumer_tasks",
                    )
        for index, task in enumerate(self.tasks):
            for value_id in task.read_values:
                if task.id not in local_value_index[value_id].consumer_tasks:
                    raise SchemaError("read value is missing task from consumers", path=f"{path}.tasks[{index}].read_values")
            for value_id in task.write_values:
                if task.id not in local_value_index[value_id].producer_tasks:
                    raise SchemaError("written value names a different producer", path=f"{path}.tasks[{index}].write_values")
        ancestor_cache: dict[str, set[str]] = {}

        def ancestors(task_id: str) -> set[str]:
            cached = ancestor_cache.get(task_id)
            if cached is not None:
                return cached
            result: set[str] = set()
            stack = list(task_index[task_id].deps)
            while stack:
                dependency = stack.pop()
                if dependency in result:
                    continue
                result.add(dependency)
                stack.extend(task_index[dependency].deps)
            ancestor_cache[task_id] = result
            return result

        for index, task in enumerate(self.tasks):
            dependency_closure = ancestors(task.id)
            for value_id in task.read_values:
                value = local_value_index[value_id]
                producer_tasks = tuple(task_index[item] for item in value.producer_tasks)
                relevant_producers = {
                    producer.id
                    for producer in producer_tasks
                    if task.tensor_slice is None
                    or producer.tensor_slice is None
                    or len(task.tensor_slice.shape) != len(producer.tensor_slice.shape)
                    or _slices_overlap(task.tensor_slice, producer.tensor_slice)
                }
                if not relevant_producers.issubset(dependency_closure):
                    raise SchemaError(
                        "consumer dependency closure omits a relevant slice producer",
                        path=f"{path}.tasks[{index}].deps",
                    )
        staging_by_access: dict[str, list[StateStagingValue]] = {}
        for value in self.state_staging_values:
            staging_by_access.setdefault(value.state_access_ref, []).append(value)
        state_tasks_by_access: dict[str, list[SemanticTask]] = {}
        for task in state_tasks:
            state_tasks_by_access.setdefault(
                task.origin_ref.state_access_ref, []
            ).append(task)
        if set(state_tasks_by_access) != set(self.state_access_ids):
            raise SchemaError(
                "StateIoOrigin accesses must exactly match state_access_ids",
                path=f"{path}.state_access_ids",
            )
        task_order = {task.id: index for index, task in enumerate(self.tasks)}
        for access_index, access_id in enumerate(self.state_access_ids):
            staging_values = staging_by_access.get(access_id, [])
            if len(staging_values) != 1:
                raise SchemaError(
                    "each local state access requires exactly one staging value",
                    path=f"{path}.state_access_ids[{access_index}]",
                )
            staging_value = staging_values[0]
            access_dma_tasks = state_tasks_by_access.get(access_id, [])
            if not access_dma_tasks:
                raise SchemaError(
                    "state access has no DMA task",
                    path=f"{path}.state_access_ids[{access_index}]",
                )
            kinds = tuple(task.kind for task in access_dma_tasks)
            if len(set(kinds)) != len(kinds):
                raise SchemaError(
                    "state access permits at most one DMA_IN and one DMA_OUT",
                    path=f"{path}.state_access_ids[{access_index}]",
                )
            first_contract = access_dma_tasks[0].dma
            assert first_contract is not None
            access_task_refs = first_contract.access_task_refs
            if tuple(
                sorted(access_task_refs, key=task_order.__getitem__)
            ) != access_task_refs:
                raise SchemaError(
                    "access_task_refs must follow canonical local task order",
                    path=f"{path}.state_access_ids[{access_index}]",
                )
            for target_ref in access_task_refs:
                target = task_index.get(target_ref)
                if target is None or isinstance(target.origin_ref, StateIoOrigin):
                    raise SchemaError(
                        "access_task_refs must name local non-DMA tasks",
                        path=f"{path}.state_access_ids[{access_index}]",
                    )
            for dma_task in access_dma_tasks:
                origin = dma_task.origin_ref
                contract = dma_task.dma
                assert isinstance(origin, StateIoOrigin)
                assert contract is not None
                if (
                    contract.state_ref != staging_value.state_ref
                    or contract.local_value_ref != staging_value.id
                    or contract.access_task_refs != access_task_refs
                    or dma_task.tensor_slice is None
                    or dma_task.tensor_slice.value_id != staging_value.id
                    or dma_task.tensor_slice.shape != dma_task.shape
                    or len(dma_task.shape) != len(staging_value.shape)
                    or any(
                        dma_task.tensor_slice.offset[axis]
                        + dma_task.tensor_slice.shape[axis]
                        > staging_value.shape[axis]
                        for axis in range(len(dma_task.shape))
                    )
                    or dma_task.dtype is not staging_value.dtype
                ):
                    raise SchemaError(
                        "DMA contract/payload disagrees with its staging value",
                        path=f"{path}.tasks[{task_order[dma_task.id]}]",
                    )
                for target_ref in access_task_refs:
                    target = task_index[target_ref]
                    target_rank = getattr(target.origin_ref, "rank", None)
                    target_node = (
                        target.origin_ref.op_id
                        if isinstance(target.origin_ref, OrdinaryNodeOrigin)
                        else target.member_id
                    )
                    if target_rank != origin.rank or (
                        target_node is not None
                        and target_node != origin.node_ref
                    ):
                        raise SchemaError(
                            "state access target disagrees with origin node/rank",
                            path=f"{path}.tasks[{task_order[dma_task.id]}].origin_ref",
                        )
                    if (
                        dma_task.kind is SemanticTaskKind.DMA_IN
                        and dma_task.id not in ancestors(target_ref)
                    ):
                        raise SchemaError(
                            "DMA_IN must precede every state access target",
                            path=f"{path}.tasks[{task_order[target_ref]}].deps",
                        )
                    if (
                        dma_task.kind is SemanticTaskKind.DMA_OUT
                        and target_ref not in ancestors(dma_task.id)
                    ):
                        raise SchemaError(
                            "DMA_OUT must follow every state access target",
                            path=f"{path}.tasks[{task_order[dma_task.id]}].deps",
                        )
        for index, flow in enumerate(self.flows):
            if not set(flow.task_ids).issubset(task_index):
                raise SchemaError("flow contains dangling task", path=f"{path}.flows[{index}].task_ids")
            expected = {task.id for task in self.tasks if task.flow_id == flow.id}
            if set(flow.task_ids) != expected:
                raise SchemaError("flow task_ids disagree with task flow_id", path=f"{path}.flows[{index}].task_ids")
            is_state_transfer_flow = bool(flow.task_ids) and all(
                isinstance(task_index[task_id].origin_ref, StateTransferOrigin)
                for task_id in flow.task_ids
            )
            flow.validate(
                f"{path}.flows[{index}]",
                allow_equal_ranks_for_state_transfer=is_state_transfer_flow,
            )
            for task_id in flow.task_ids:
                task = task_index[task_id]
                if isinstance(task.origin_ref, StateTransferOrigin):
                    origin = task.origin_ref
                    if (
                        flow.tensor_slice.value_id
                        != canonical_state_transfer_payload_id(
                            origin.state_transfer_ref,
                            origin.segment_index,
                        )
                        or task.tensor_slice is None
                        or (
                            origin.segment_index is None
                            and task.tensor_slice.offset != flow.tensor_slice.offset
                        )
                        or task.tensor_slice.shape != flow.tensor_slice.shape
                        or task.bytes != flow.bytes
                        or task.dtype != flow.dtype
                    ):
                        raise SchemaError(
                            "state transfer logical/local payload geometry disagrees",
                            path=f"{path}.flows[{index}]",
                        )
                    if task.kind is SemanticTaskKind.SEND:
                        endpoint_refs = task.read_values
                        forbidden_refs = task.write_values
                    elif task.kind is SemanticTaskKind.RECV:
                        endpoint_refs = task.write_values
                        forbidden_refs = task.read_values
                    else:
                        endpoint_refs = ()
                        forbidden_refs = task.read_values + task.write_values
                    if (
                        forbidden_refs
                        or (
                            task.kind
                            in (SemanticTaskKind.SEND, SemanticTaskKind.RECV)
                            and (
                                len(endpoint_refs) != 1
                                or endpoint_refs[0] not in staging_index
                                or task.tensor_slice.value_id != endpoint_refs[0]
                            )
                        )
                        or (
                            task.kind is SemanticTaskKind.TRANSIT
                            and task.tensor_slice != flow.tensor_slice
                        )
                    ):
                        raise SchemaError(
                            "state transfer endpoint must map the logical payload to exactly one local staging value",
                            path=f"{path}.flows[{index}]",
                        )
                elif (
                    task.tensor_slice != flow.tensor_slice
                    or task.bytes != flow.bytes
                    or task.dtype != flow.dtype
                ):
                    raise SchemaError(
                        "flow payload disagrees with task",
                        path=f"{path}.flows[{index}]",
                    )
                if task.kind not in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                    SemanticTaskKind.TRANSIT,
                ):
                    raise SchemaError("flow contains a non-transport task", path=f"{path}.flows[{index}]")
            if self.die_id not in flow.die_path:
                raise SchemaError("flow is attached to a die outside its selected path", path=f"{path}.flows[{index}].die_path")
        for index, region in enumerate(self.regions):
            region.validate(f"{path}.regions[{index}]")
            if not set(region.task_ids).issubset(task_index):
                raise SchemaError("region contains dangling task", path=f"{path}.regions[{index}].task_ids")
            expected_tasks = tuple(
                task for task in self.tasks if task.region_id == region.id
            )
            if region.task_ids != tuple(task.id for task in expected_tasks):
                raise SchemaError(
                    "region task_ids must exactly follow local task order",
                    path=f"{path}.regions[{index}].task_ids",
                )
            if region.fusion_plan_id is not None and region.fusion_plan_id not in self.fusion_plan_ids:
                raise SchemaError("dangling fusion plan", path=f"{path}.regions[{index}].fusion_plan_id")
            if region.standalone_collective_plan_id is not None and region.standalone_collective_plan_id not in self.standalone_collective_plan_ids:
                raise SchemaError("dangling standalone plan", path=f"{path}.regions[{index}].standalone_collective_plan_id")
            if region.lowering is RegionLowering.ISA_REGION and any(
                not isinstance(task.origin_ref, FusedNodeOrigin)
                or task.origin_ref.plan_id != region.fusion_plan_id
                for task in expected_tasks
            ):
                raise SchemaError(
                    "ISA region tasks must all have fused origins for the same plan",
                    path=f"{path}.regions[{index}]",
                )
            if region.lowering is RegionLowering.STRICT_ACTIONS and any(
                not isinstance(task.origin_ref, StandaloneNodeOrigin)
                or task.origin_ref.collective_plan_id
                != region.standalone_collective_plan_id
                for task in expected_tasks
            ):
                raise SchemaError(
                    "strict-actions region tasks must all have standalone origins for the same plan",
                    path=f"{path}.regions[{index}]",
                )
            if region.lowering is RegionLowering.STRICT_STATE_IO:
                if (
                    len(expected_tasks) != 1
                    or not isinstance(expected_tasks[0].origin_ref, StateIoOrigin)
                    or expected_tasks[0].kind
                    not in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT)
                ):
                    raise SchemaError(
                        "strict-state-io region must contain exactly one state DMA",
                        path=f"{path}.regions[{index}]",
                    )
                state_task = expected_tasks[0]
                state_origin = state_task.origin_ref
                assert isinstance(state_origin, StateIoOrigin)
                if region.id != canonical_state_region_id(
                    state_origin.state_access_ref, state_task.kind
                ) or region.task_ids != (state_task.id,):
                    raise SchemaError(
                        "strict-state-io region identity/partition is not canonical",
                        path=f"{path}.regions[{index}]",
                    )
            if region.lowering is RegionLowering.STRICT_STATE_TRANSFER:
                transfer_ref = region.state_transfer_ref
                assert transfer_ref is not None
                origins = tuple(
                    task.origin_ref for task in expected_tasks
                )
                kinds = tuple(task.kind for task in expected_tasks)
                segment_indices = tuple(
                    getattr(origin, "segment_index", None)
                    for origin in origins
                )
                segment_pairs = tuple(zip(kinds, segment_indices))
                legacy_shape = (
                    kinds
                    in (
                        (SemanticTaskKind.SEND,),
                        (SemanticTaskKind.RECV, SemanticTaskKind.WAIT),
                        (SemanticTaskKind.TRANSIT,),
                    )
                    and all(index is None for index in segment_indices)
                )
                segmented_shape = (
                    all(type(index) is int for index in segment_indices)
                    and (
                        segment_pairs
                        == tuple(
                            (SemanticTaskKind.SEND, index)
                            for index in range(len(expected_tasks))
                        )
                        or segment_pairs
                        == tuple(
                            item
                            for index in range(len(expected_tasks) // 2)
                            for item in (
                                (SemanticTaskKind.RECV, index),
                                (SemanticTaskKind.WAIT, index),
                            )
                        )
                        or segment_pairs
                        == tuple(
                            (SemanticTaskKind.TRANSIT, index)
                            for index in range(len(expected_tasks))
                        )
                    )
                )
                if (
                    transfer_ref not in self.state_transfer_ids
                    or any(
                        not isinstance(origin, StateTransferOrigin)
                        or origin.state_transfer_ref != transfer_ref
                        for origin in origins
                    )
                    or not (legacy_shape or segmented_shape)
                    or region.id
                    != canonical_state_transfer_region_id(
                        transfer_ref, self.die_id
                    )
                    or region.task_ids
                    != tuple(
                        canonical_state_transfer_task_id(
                            transfer_ref, task.kind, self.die_id,
                            origin.segment_index,
                        )
                        for task, origin in zip(
                            expected_tasks, origins, strict=True
                        )
                    )
                ):
                    raise SchemaError(
                        "strict-state-transfer identity/partition is not canonical",
                        path=f"{path}.regions[{index}]",
                    )
            if region.lowering is RegionLowering.JSON_COARSE and any(
                not isinstance(task.origin_ref, OrdinaryNodeOrigin)
                for task in expected_tasks
            ):
                raise SchemaError(
                    "JSON coarse region tasks must all have ordinary origins",
                    path=f"{path}.regions[{index}]",
                )
        fusion_regions = tuple(
            region.fusion_plan_id
            for region in self.regions
            if region.lowering is RegionLowering.ISA_REGION
        )
        if fusion_regions != self.fusion_plan_ids:
            raise SchemaError(
                "each local fusion plan must have exactly one ISA region in plan order",
                path=f"{path}.regions",
            )
        standalone_regions = tuple(
            region.standalone_collective_plan_id
            for region in self.regions
            if region.lowering is RegionLowering.STRICT_ACTIONS
        )
        if standalone_regions != self.standalone_collective_plan_ids:
            raise SchemaError(
                "each local standalone plan must have exactly one strict-actions region in plan order",
                path=f"{path}.regions",
            )
        expected_id = stable_artifact_id("intra_die_dag", self._semantic_key(), schema_version=INTRA_DIE_DAG_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        fusion_plans: tuple[FusionPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
        path: str = "intra_die_dag",
    ) -> None:
        self.validate(path)
        task_index = {task.id: task for task in self.tasks}
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("DAG references a different IR-1", path=f"{path}.source_ir1_id")
        if self.die_id not in {die.id for die in ir1.fabric.dies}:
            raise SchemaError("DAG references an unknown die", path=f"{path}.die_id")
        manifest = ir1.persistent_state_manifest
        expected_manifest_id = manifest.id if manifest is not None else None
        if self.source_state_manifest_id != expected_manifest_id:
            raise SchemaError(
                "DAG persistent-state manifest provenance disagrees with IR-1",
                path=f"{path}.source_state_manifest_id",
            )
        if manifest is None:
            expected_local_accesses = ()
        else:
            binding_die = {
                binding.state_ref: binding.die_id
                for binding in manifest.bindings
            }
            expected_local_accesses = tuple(
                access
                for access in ir1.state_accesses
                if binding_die[access.state_ref] == self.die_id
            )
        if self.state_access_ids != tuple(
            access.id for access in expected_local_accesses
        ):
            raise SchemaError(
                "state_access_ids must exactly cover IR-1 accesses homed on this die",
                path=f"{path}.state_access_ids",
            )
        if manifest is not None:
            declarations = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }
            staging_by_access = {
                value.state_access_ref: value
                for value in self.state_staging_values
            }
            state_tasks_by_access: dict[str, list[SemanticTask]] = {}
            for task in self.tasks:
                if isinstance(task.origin_ref, StateIoOrigin):
                    state_tasks_by_access.setdefault(
                        task.origin_ref.state_access_ref, []
                    ).append(task)
            expected_kinds_by_mode = {
                StateAccessMode.READ: (SemanticTaskKind.DMA_IN,),
                StateAccessMode.WRITE: (SemanticTaskKind.DMA_OUT,),
                StateAccessMode.READ_WRITE: (
                    SemanticTaskKind.DMA_IN,
                    SemanticTaskKind.DMA_OUT,
                ),
            }
            for access_index, access in enumerate(expected_local_accesses):
                declaration = declarations[access.state_ref]
                expected_kinds = expected_kinds_by_mode[access.mode]
                (
                    staging_shape,
                    staging_layout,
                    dma_offset,
                    dma_shape,
                ) = canonical_state_access_view(
                    ir1,
                    access,
                    declaration,
                    expected_kinds[0],
                )

                dma_tasks = state_tasks_by_access.get(access.id, [])
                if tuple(task.kind for task in dma_tasks) != expected_kinds:
                    raise SchemaError(
                        "DMA_IN/DMA_OUT coverage disagrees with IR-1 access mode",
                        path=f"{path}.state_access_ids[{access_index}]",
                    )
                staging = staging_by_access.get(access.id)
                if staging is None or (
                    staging.state_ref,
                    staging.shape,
                    staging.dtype,
                    staging.logical_layout,
                ) != (
                    declaration.id,
                    staging_shape,
                    declaration.dtype,
                    staging_layout,
                ):
                    raise SchemaError(
                        "state access requires one matching canonical staging domain",
                        path=f"{path}.state_access_ids[{access_index}]",
                    )
                root_slice = TensorSlice(
                    staging.id,
                    (0,) * len(staging_shape),
                    staging_shape,
                )
                first_contract = dma_tasks[0].dma
                assert first_contract is not None
                expected_access_task_refs = tuple(
                    task.id
                    for task in self.tasks
                    if task.kind is SemanticTaskKind.COMP
                    and task.member_id == access.node_ref
                    and getattr(task.origin_ref, "rank", None) == access.rank
                )
                if (
                    first_contract.access_task_refs
                    != expected_access_task_refs
                ):
                    raise SchemaError(
                        "access_task_refs must exactly cover IR-1 node/rank compute tasks",
                        path=f"{path}.state_access_ids[{access_index}]",
                    )
                if declaration.identity.tensor_ref is not None and any(
                    staging.id not in task_index[target_ref].read_values
                    or task_index[target_ref].compute is None
                    or staging.id
                    not in tuple(
                        operand.value_id
                        for operand in task_index[target_ref].compute.inputs
                    )
                    for target_ref in first_contract.access_task_refs
                ):
                    raise SchemaError(
                        "state access target compute must consume the staging value",
                        path=f"{path}.state_access_ids[{access_index}]",
                    )
                if declaration.identity.tensor_ref is not None:
                    for target_ref in first_contract.access_task_refs:
                        target = task_index[target_ref]
                        if (
                            target.compute is None
                            or target.compute.tile is None
                        ):
                            continue
                        tile_slices = tuple(
                            binding
                            for binding in target.compute.tile.input_slices
                            if binding.operand_id == staging.id
                        )
                        if len(tile_slices) != 1 or (
                            tile_slices[0].source_value_id,
                            tile_slices[0].logical_offset,
                            tile_slices[0].logical_shape,
                        ) != (
                            declaration.identity.tensor_ref,
                            dma_offset,
                            dma_shape,
                        ):
                            raise SchemaError(
                                "parameter compute tile disagrees with canonical state shard view",
                                path=(
                                    f"{path}.state_access_ids"
                                    f"[{access_index}]"
                                ),
                            )
                for dma_task in dma_tasks:
                    origin = dma_task.origin_ref
                    contract = dma_task.dma
                    assert isinstance(origin, StateIoOrigin)
                    assert contract is not None
                    (
                        _staging_shape,
                        _staging_layout,
                        task_offset,
                        task_shape,
                    ) = canonical_state_access_view(
                        ir1,
                        access,
                        declaration,
                        dma_task.kind,
                    )
                    expected_slice = TensorSlice(
                        staging.id,
                        task_offset,
                        task_shape,
                    )
                    element_bytes = declaration.tensor_bytes // math.prod(
                        declaration.shape
                    )
                    expected_bytes = math.prod(task_shape) * element_bytes
                    state_offset_bytes = (
                        0
                        if declaration.identity.tensor_ref is not None
                        else dense_row_major_view_byte_addend(
                            root_slice,
                            expected_slice,
                            declaration.dtype,
                            path=(
                                f"{path}.state_access_ids"
                                f"[{access_index}]"
                            ),
                        )
                    )
                    if (
                        dma_task.tensor_slice != expected_slice
                        or dma_task.bytes != expected_bytes
                        or dma_task.dtype is not declaration.dtype
                        or dma_task.shape != task_shape
                        or contract.state_offset_bytes != state_offset_bytes
                        or contract.access_task_refs
                        != expected_access_task_refs
                    ):
                        raise SchemaError(
                            "state DMA payload disagrees with canonical direction view",
                            path=f"{path}.state_access_ids[{access_index}]",
                        )
                    if (
                        origin.node_ref,
                        origin.rank,
                        contract.state_ref,
                    ) != (
                        access.node_ref,
                        access.rank,
                        access.state_ref,
                    ):
                        raise SchemaError(
                            "state DMA provenance disagrees with IR-1 access",
                            path=f"{path}.state_access_ids[{access_index}]",
                        )
        fusion_index = {plan.id: plan for plan in fusion_plans}
        standalone_index = {plan.id: plan for plan in standalone_plans}
        if len(fusion_index) != len(fusion_plans) or not set(self.fusion_plan_ids).issubset(fusion_index):
            raise SchemaError("fusion plan inputs are duplicate or missing declared ids", path=f"{path}.fusion_plan_ids")
        if len(standalone_index) != len(standalone_plans) or not set(self.standalone_collective_plan_ids).issubset(standalone_index):
            raise SchemaError("standalone plan inputs are duplicate or missing declared ids", path=f"{path}.standalone_collective_plan_ids")
        for plan_id in self.fusion_plan_ids:
            fusion_index[plan_id].validate_against(ir1)
        for plan_id in self.standalone_collective_plan_ids:
            standalone_index[plan_id].validate_against(ir1)
        ir1_node_index = {node.id: node for node in ir1.nodes}
        ir1_nodes = set(ir1_node_index)
        if not set(self.ordinary_node_ids).issubset(ir1_nodes):
            raise SchemaError("ordinary origins contain a dangling IR-1 node", path=f"{path}.ordinary_node_ids")
        local_ordinary_ids = {
            task.origin_ref.op_id
            for task in self.tasks
            if isinstance(task.origin_ref, OrdinaryNodeOrigin)
        }
        if set(self.ordinary_node_ids) != local_ordinary_ids:
            raise SchemaError(
                "ordinary_node_ids must exactly match local ordinary origins",
                path=f"{path}.ordinary_node_ids",
            )
        fusion_actions = {
            (plan.id, program.rank, action.id): action
            for plan in fusion_plans
            for program in plan.rank_programs
            for action in program.actions
        }
        standalone_actions = {
            (plan.id, program.rank, action.id): action
            for plan in standalone_plans
            for program in plan.rank_programs
            for action in program.actions
        }
        fusion_plan_index = {plan.id: plan for plan in fusion_plans}
        standalone_plan_index = {plan.id: plan for plan in standalone_plans}
        groups = {group.id: group for group in ir1.groups}
        values = {value.id: value for value in ir1.values}
        planned_temp_origins: dict[str, str] = {}

        def bind_temp_origin(temp_id: str, origin_value_id: str) -> None:
            if temp_id in values:
                if temp_id != origin_value_id:
                    raise SchemaError(
                        "planned operand aliases an unrelated IR-1 value id",
                        path=f"{path}.values",
                    )
                return
            previous = planned_temp_origins.get(temp_id)
            if previous is not None and previous != origin_value_id:
                raise SchemaError(
                    "planned temp id has conflicting IR-1 origins",
                    path=f"{path}.values",
                )
            planned_temp_origins[temp_id] = origin_value_id

        skeleton_index = {
            skeleton.id: skeleton for skeleton in ir1.fused_op_skeletons
        }
        for plan in fusion_plans:
            skeleton = skeleton_index[plan.fused_op_id]
            gemm = ir1_node_index[skeleton.member_node_ids[0]]
            partial_value_id = gemm.outputs[0]
            for program in plan.rank_programs:
                for action in program.actions:
                    if action.compute is not None and action.compute.tile is not None:
                        for binding in (
                            action.compute.tile.input_slices
                            + action.compute.tile.output_slices
                        ):
                            bind_temp_origin(
                                binding.operand_id,
                                binding.source_value_id,
                            )
                    if action.kind is FusionActionKind.RECV:
                        for temp_id in action.writes:
                            bind_temp_origin(temp_id, partial_value_id)
        local_origin_tasks: dict[tuple[str, int, str], SemanticTask] = {}
        for task in self.tasks:
            if task.kind is SemanticTaskKind.TRANSIT:
                continue
            origin = task.origin_ref
            if isinstance(origin, FusedNodeOrigin):
                local_origin_tasks[(origin.plan_id, origin.rank, origin.action_id)] = task
            elif isinstance(origin, StandaloneNodeOrigin):
                local_origin_tasks[(origin.collective_plan_id, origin.rank, origin.action_id)] = task
        skeletons = {skeleton.id: skeleton for skeleton in ir1.fused_op_skeletons}
        fusion_node_units: dict[str, str] = {}
        for plan in fusion_plans:
            skeleton = skeletons[plan.fused_op_id]
            for member_id in skeleton.member_node_ids:
                fusion_node_units[member_id] = plan.id
        standalone_node_units = {plan.op_id: plan.id for plan in standalone_plans}

        def coverage_unit(node_id: str) -> tuple[str, str]:
            if node_id in fusion_node_units:
                return ("fusion", fusion_node_units[node_id])
            if node_id in standalone_node_units:
                return ("standalone", standalone_node_units[node_id])
            return ("ordinary", node_id)

        def belongs_to_node(task: SemanticTask, node_id: str) -> bool:
            origin = task.origin_ref
            unit_kind, unit_id = coverage_unit(node_id)
            if unit_kind == "ordinary":
                return (
                    isinstance(origin, OrdinaryNodeOrigin)
                    and origin.op_id == node_id
                )
            if unit_kind == "fusion":
                return (
                    isinstance(origin, FusedNodeOrigin)
                    and origin.plan_id == unit_id
                    and task.member_id == node_id
                )
            return (
                isinstance(origin, StandaloneNodeOrigin)
                and origin.collective_plan_id == unit_id
                and task.member_id == node_id
            )

        def belongs_to_unit(task: SemanticTask, node_id: str) -> bool:
            origin = task.origin_ref
            unit_kind, unit_id = coverage_unit(node_id)
            if unit_kind == "ordinary":
                return (
                    isinstance(origin, OrdinaryNodeOrigin)
                    and origin.op_id == unit_id
                )
            if unit_kind == "fusion":
                return (
                    isinstance(origin, FusedNodeOrigin)
                    and origin.plan_id == unit_id
                )
            return (
                isinstance(origin, StandaloneNodeOrigin)
                and origin.collective_plan_id == unit_id
            )

        def entry_tasks(
            node_id: str,
            value_id: str | None,
            *,
            whole_unit: bool,
        ) -> tuple[SemanticTask, ...]:
            unit_kind, _unit_id = coverage_unit(node_id)
            expected_kind = {
                "ordinary": SemanticTaskKind.COMP,
                "fusion": SemanticTaskKind.COMP,
                "standalone": SemanticTaskKind.LOCAL_COPY,
            }[unit_kind]

            def reads_origin_value(task: SemanticTask) -> bool:
                if value_id is None or value_id in task.read_values:
                    return True
                if (
                    unit_kind == "fusion"
                    and task.kind is SemanticTaskKind.COMP
                    and task.compute is not None
                    and task.compute.tile is not None
                ):
                    return any(
                        binding.source_value_id == value_id
                        and binding.operand_id in task.read_values
                        for binding in task.compute.tile.input_slices
                    )
                return False

            return tuple(
                sorted(
                    (
                        task
                        for task in self.tasks
                        if (
                            belongs_to_unit(task, node_id)
                            if whole_unit
                            else belongs_to_node(task, node_id)
                        )
                        and task.kind is expected_kind
                        and reads_origin_value(task)
                    ),
                    key=lambda task: task.id,
                )
            )

        def completion_tasks(
            node_id: str,
            value_id: str | None,
            *,
            whole_unit: bool,
        ) -> tuple[SemanticTask, ...]:
            unit_kind, _unit_id = coverage_unit(node_id)
            expected_kind = {
                "ordinary": SemanticTaskKind.COMP,
                "fusion": SemanticTaskKind.REDUCE,
                "standalone": SemanticTaskKind.BARRIER,
            }[unit_kind]
            return tuple(
                sorted(
                    (
                        task
                        for task in self.tasks
                        if (
                            belongs_to_unit(task, node_id)
                            if whole_unit
                            else belongs_to_node(task, node_id)
                        )
                        and task.kind is expected_kind
                        and (
                            value_id is None
                            or unit_kind == "standalone"
                            or value_id in task.write_values
                        )
                    ),
                    key=lambda task: task.id,
                )
            )

        graph_dep_lists: dict[str, list[str]] = {
            task.id: [] for task in self.tasks
        }
        for edge in sorted(ir1.edges, key=lambda edge: edge.id):
            if (
                ir1_node_index[edge.source_node].instance_id
                != ir1_node_index[edge.destination_node].instance_id
                and self.state_transfer_ids
            ):
                # Cross-instance sequencing is represented by the exact
                # SEND/RECV/WAIT quotient validated by the parent projection.
                continue
            if coverage_unit(edge.source_node) == coverage_unit(edge.destination_node):
                continue
            whole_unit = edge.kind is EdgeKind.CONTROL
            value_id = edge.value_id if not whole_unit else None
            entries = entry_tasks(
                edge.destination_node,
                value_id,
                whole_unit=whole_unit,
            )
            if not entries:
                continue
            completions = completion_tasks(
                edge.source_node,
                value_id,
                whole_unit=whole_unit,
            )
            if not completions:
                raise SchemaError(
                    "cross-coverage dependency has no completion on the entry task die",
                    path=f"{path}.tasks",
                )
            for entry in entries:
                graph_dep_lists[entry.id].extend(
                    completion.id for completion in completions
                )
        graph_deps = {
            task_id: tuple(dict.fromkeys(dependencies))
            for task_id, dependencies in graph_dep_lists.items()
        }
        state_in_deps: dict[str, list[str]] = {
            task.id: [] for task in self.tasks
        }
        for state_task in self.tasks:
            if (
                state_task.kind is SemanticTaskKind.DMA_IN
                and state_task.dma is not None
            ):
                for target_ref in state_task.dma.access_task_refs:
                    state_in_deps[target_ref].append(state_task.id)
        transfer_wait_deps: dict[str, tuple[str, ...]] = {
            task.id: tuple(
                dependency
                for dependency in task.deps
                if (
                    dependency in task_index
                    and isinstance(
                        task_index[dependency].origin_ref,
                        StateTransferOrigin,
                    )
                    and task_index[dependency].kind is SemanticTaskKind.WAIT
                )
            )
            for task in self.tasks
        }
        for index, task in enumerate(self.tasks):
            origin = task.origin_ref
            if isinstance(origin, FusedNodeOrigin):
                source_action = fusion_actions.get((origin.plan_id, origin.rank, origin.action_id))
                source_plan = fusion_plan_index.get(origin.plan_id)
                if source_action is None:
                    raise SchemaError("fused origin references a dangling action", path=f"{path}.tasks[{index}].origin_ref")
            elif isinstance(origin, StandaloneNodeOrigin):
                source_action = standalone_actions.get((origin.collective_plan_id, origin.rank, origin.action_id))
                source_plan = standalone_plan_index.get(origin.collective_plan_id)
                if source_action is None:
                    raise SchemaError("standalone origin references a dangling action", path=f"{path}.tasks[{index}].origin_ref")
            elif isinstance(origin, (StateIoOrigin, StateTransferOrigin)):
                # State I/O and transfer tasks are validated against their
                # independent provenance contracts below.
                continue
            else:
                source_action = None
                source_plan = None
                if origin.op_id not in ir1_nodes:
                    raise SchemaError("ordinary origin references a dangling node", path=f"{path}.tasks[{index}].origin_ref")
            if source_action is not None:
                if task.kind is SemanticTaskKind.TRANSIT:
                    if source_action.kind is not FusionActionKind.SEND:
                        raise SchemaError("TRANSIT must derive from SEND", path=f"{path}.tasks[{index}].origin_ref")
                    continue
                if task.kind.value != source_action.kind.value:
                    raise SchemaError("task kind disagrees with originating action", path=f"{path}.tasks[{index}].kind")
                assert source_plan is not None
                group = groups[source_plan.group_ref]
                expected_die = next(
                    placement.die_id
                    for placement in group.placements
                    if placement.rank == origin.rank
                )
                if self.die_id != expected_die:
                    raise SchemaError("action is projected onto the wrong die", path=f"{path}.tasks[{index}]")
                chunk = next(
                    (item for item in source_plan.chunk_slices if item.id == source_action.slice_ref),
                    None,
                )
                expected_slice = (
                    TensorSlice(chunk.value_id, chunk.offset, chunk.shape)
                    if chunk is not None
                    else None
                )
                if (
                    task.member_id,
                    task.chunk_id,
                    task.collective_step,
                    task.tensor_slice,
                    task.bytes,
                    task.dtype,
                    task.read_values,
                    task.write_values,
                    task.compute,
                    task.reduction,
                    task.sync,
                ) != (
                    source_action.member_id,
                    source_action.chunk_id,
                    source_action.collective_step,
                    expected_slice,
                    source_action.bytes,
                    source_action.dtype,
                    source_action.reads,
                    source_action.writes,
                    source_action.compute,
                    source_action.reduction,
                    source_action.sync,
                ):
                    raise SchemaError(
                        "task fields disagree with originating action",
                        path=f"{path}.tasks[{index}]",
                    )
                expected_op_kind = (
                    source_action.compute.op_kind
                    if source_action.compute is not None
                    else OpKind.COLLECTIVE
                )
                if task.op_kind is not expected_op_kind:
                    raise SchemaError("op_kind disagrees with originating action", path=f"{path}.tasks[{index}].op_kind")
                expected_deps = tuple(
                    local_origin_tasks[(
                        origin.plan_id if isinstance(origin, FusedNodeOrigin) else origin.collective_plan_id,
                        origin.rank,
                        dependency,
                    )].id
                    for dependency in source_action.deps
                )
                expected_deps = tuple(
                    dict.fromkeys(
                        expected_deps
                        + tuple(state_in_deps[task.id])
                        + transfer_wait_deps[task.id]
                        + graph_deps[task.id]
                    )
                )
                if task.deps != expected_deps:
                    raise SchemaError(
                        "deps disagree with internal plus canonical graph dependencies",
                        path=f"{path}.tasks[{index}].deps",
                    )
                if source_action.kind in (FusionActionKind.SEND, FusionActionKind.RECV):
                    flow = next(item for item in self.flows if item.id == task.flow_id)
                    source_rank, destination_rank = (
                        (origin.rank, source_action.peer_rank)
                        if source_action.kind is FusionActionKind.SEND
                        else (source_action.peer_rank, origin.rank)
                    )
                    route = next(
                        candidate
                        for candidate in group.embedding.routes
                        if candidate.source_rank == source_rank
                        and candidate.destination_rank == destination_rank
                        and candidate.die_path == source_action.expected_route
                    )
                    if (
                        task.source_rank,
                        task.destination_rank,
                        flow.logical_channel,
                        flow.pair_route_ref,
                        flow.source_rank,
                        flow.destination_rank,
                        flow.die_path,
                    ) != (
                        source_rank,
                        destination_rank,
                        source_action.logical_channel,
                        route.id,
                        source_rank,
                        destination_rank,
                        source_action.expected_route,
                    ):
                        raise SchemaError("transport identity disagrees with action/route", path=f"{path}.tasks[{index}]")
                elif task.flow_id is not None:
                    raise SchemaError("non-transport action cannot reference a flow", path=f"{path}.tasks[{index}].flow_id")
            else:
                node = ir1_node_index[origin.op_id]
                group = groups[node.execution_group_ref]
                placement = next(
                    (
                        placement
                        for placement in group.placements
                        if placement.rank == origin.rank
                    ),
                    None,
                )
                if placement is None:
                    raise SchemaError(
                        "ordinary origin rank is absent from its execution group",
                        path=f"{path}.tasks[{index}].origin_ref.rank",
                    )
                if placement.die_id != self.die_id:
                    raise SchemaError(
                        "ordinary task is projected onto the wrong die",
                        path=f"{path}.tasks[{index}]",
                    )
                if task.kind is not SemanticTaskKind.COMP or task.compute is None:
                    raise SchemaError(
                        "ordinary origin requires one self-contained COMP task",
                        path=f"{path}.tasks[{index}]",
                    )
                compute = task.compute
                expected_inputs = list(node.inputs)
                for state_access in expected_local_accesses:
                    if (
                        state_access.node_ref != node.id
                        or state_access.rank != origin.rank
                    ):
                        continue
                    declaration = declarations[state_access.state_ref]
                    tensor_ref = declaration.identity.tensor_ref
                    if tensor_ref is None:
                        continue
                    matching_inputs = tuple(
                        input_index
                        for input_index, value_id in enumerate(expected_inputs)
                        if value_id == tensor_ref
                    )
                    if len(matching_inputs) != 1:
                        raise SchemaError(
                            "parameter access does not identify one PhysicalNode input",
                            path=f"{path}.tasks[{index}]",
                        )
                    expected_inputs[matching_inputs[0]] = staging_by_access[
                        state_access.id
                    ].id
                expected_inputs_tuple = tuple(expected_inputs)
                input_roles, output_roles = canonical_compute_operand_roles(
                    node.kind,
                    node.workload,
                    tiled=False,
                    path=f"{path}.tasks[{index}].compute",
                )
                if (
                    len(expected_inputs_tuple) != len(input_roles)
                    or len(node.outputs) != len(output_roles)
                ):
                    raise SchemaError(
                        "PhysicalNode operands do not match its exact compute carrier",
                        path=f"{path}.tasks[{index}]",
                    )
                expected_compute = ComputeContract(
                    op_kind=node.kind,
                    workload=node.workload,
                    math=node.math,
                    effects=node.effects,
                    impl_ref=node.impl_ref,
                    inputs=tuple(
                        ComputeOperand(value_id, role)
                        for value_id, role in zip(
                            expected_inputs_tuple, input_roles, strict=True
                        )
                    ),
                    outputs=tuple(
                        ComputeOperand(value_id, role)
                        for value_id, role in zip(
                            node.outputs, output_roles, strict=True
                        )
                    ),
                    tile=None,
                )
                if (
                    task.op_kind,
                    task.member_id,
                    task.read_values,
                    task.write_values,
                    compute,
                ) != (
                    node.kind,
                    node.id,
                    expected_inputs_tuple,
                    node.outputs,
                    expected_compute,
                ):
                    raise SchemaError(
                        "ordinary task compute contract disagrees with PhysicalNode",
                        path=f"{path}.tasks[{index}]",
                    )
                expected_deps = tuple(
                    dict.fromkeys(
                        tuple(state_in_deps[task.id])
                        + transfer_wait_deps[task.id]
                        + graph_deps[task.id]
                    )
                )
                if task.deps != expected_deps:
                    raise SchemaError(
                        "deps disagree with canonical graph dependencies",
                        path=f"{path}.tasks[{index}].deps",
                    )
        for index, value in enumerate(self.values):
            expected_origin_value_id = (
                value.id
                if value.id in values
                else planned_temp_origins.get(value.id)
            )
            if expected_origin_value_id is None:
                raise SchemaError(
                    "local value is neither an IR-1 value nor a planned temp",
                    path=f"{path}.values[{index}].origin_value_id",
                )
            if value.origin_value_id != expected_origin_value_id:
                raise SchemaError(
                    "local value origin must exactly match its IR-1 value or planned temp binding",
                    path=f"{path}.values[{index}].origin_value_id",
                )
            origin = values[expected_origin_value_id]
            if (
                value.shape,
                value.dtype,
                value.logical_layout,
                value.sharding,
                value.alias_set,
            ) != (
                origin.shape,
                origin.dtype,
                origin.logical_layout,
                origin.sharding,
                origin.alias_set,
            ):
                raise SchemaError(
                    "local value metadata disagrees with IR-1 value",
                    path=f"{path}.values[{index}]",
                )


@dataclass(frozen=True, slots=True)
class IR2ProjectionResult:
    """Complete per-die projection; this is the one-time coverage validation boundary."""

    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    fusion_plan_ids: tuple[str, ...]
    standalone_collective_plan_ids: tuple[str, ...]
    dags: tuple[IntraDieDAG, ...]
    source_state_manifest_id: str | None = None
    state_transfers: tuple[StateTransferLike, ...] = ()

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "IR2ProjectionResult":
        semantic_key.setdefault("source_state_manifest_id", None)
        semantic_key.setdefault("state_transfers", ())
        return cls(
            schema_version=IR2_PROJECTION_RESULT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "ir2_projection_result",
                semantic_key,
                schema_version=IR2_PROJECTION_RESULT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_ir1_id",
                "fusion_plan_ids",
                "standalone_collective_plan_ids",
                "dags",
                "source_state_manifest_id",
                "state_transfers",
            )
        }

    def validate(self, path: str = "ir2_projection_result") -> None:
        if self.schema_version != IR2_PROJECTION_RESULT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        if self.source_state_manifest_id is not None:
            validate_nonempty(
                self.source_state_manifest_id,
                f"{path}.source_state_manifest_id",
            )
        if type(self.state_transfers) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.state_transfers"
            )
        transfer_ids: set[str] = set()
        source_accesses: set[str] = set()
        destination_accesses: set[str] = set()
        transfer_type: type[object] | None = None
        for index, contract in enumerate(self.state_transfers):
            contract_path = f"{path}.state_transfers[{index}]"
            if type(contract) not in (
                StateTransferContract,
                SlicedKvStateTransferContract,
                SegmentedKvStateTransferContract,
            ):
                raise SchemaError(
                    "must be a StateTransferLike", path=contract_path
                )
            contract.validate(contract_path)
            if transfer_type is None:
                transfer_type = type(contract)
            elif transfer_type is not type(contract):
                raise SchemaError(
                    "one projection cannot mix state transfer contract kinds",
                    path=contract_path,
                )
            if contract.source_ir1_id != self.source_ir1_id:
                raise SchemaError(
                    "references a different IR-1",
                    path=f"{contract_path}.source_ir1_id",
                )
            duplicate_endpoint = (
                type(contract) is StateTransferContract
                and (
                    contract.source_state_access_ref in source_accesses
                    or contract.destination_state_access_ref
                    in destination_accesses
                )
            )
            if contract.id in transfer_ids or duplicate_endpoint:
                raise SchemaError(
                    "duplicate transfer id or whole-state endpoint access",
                    path=contract_path,
                )
            transfer_ids.add(contract.id)
            source_accesses.add(contract.source_state_access_ref)
            destination_accesses.add(
                contract.destination_state_access_ref
            )
        if transfer_type is StateTransferContract and self.state_transfers != tuple(
            sorted(
                self.state_transfers,
                key=lambda contract: (
                    contract.source_ir1_id,
                    contract.source_state_access_ref,
                    contract.destination_state_access_ref,
                    contract.pair_route_ref,
                    contract.id,
                ),
            )
        ):
            raise SchemaError(
                "whole-state transfers must use canonical hash order",
                path=f"{path}.state_transfers",
            )
        for field_name in ("fusion_plan_ids", "standalone_collective_plan_ids"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate plan ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")
        dag_ids: set[str] = set()
        die_ids: set[int] = set()
        for index, dag in enumerate(self.dags):
            dag.validate(f"{path}.dags[{index}]")
            if dag.id in dag_ids or dag.die_id in die_ids:
                raise SchemaError("DAG id and die_id must be unique", path=f"{path}.dags[{index}]")
            if dag.source_ir1_id != self.source_ir1_id:
                raise SchemaError("DAG references a different IR-1", path=f"{path}.dags[{index}].source_ir1_id")
            if not set(dag.fusion_plan_ids).issubset(self.fusion_plan_ids):
                raise SchemaError("DAG contains an undeclared fusion plan", path=f"{path}.dags[{index}].fusion_plan_ids")
            if not set(dag.standalone_collective_plan_ids).issubset(
                self.standalone_collective_plan_ids
            ):
                raise SchemaError("DAG contains an undeclared standalone plan", path=f"{path}.dags[{index}].standalone_collective_plan_ids")
            if dag.source_state_manifest_id != self.source_state_manifest_id:
                raise SchemaError(
                    "DAG references a different persistent-state manifest",
                    path=f"{path}.dags[{index}].source_state_manifest_id",
                )
            if not set(dag.state_transfer_ids).issubset(transfer_ids):
                raise SchemaError(
                    "DAG contains an undeclared state transfer",
                    path=f"{path}.dags[{index}].state_transfer_ids",
                )
            dag_ids.add(dag.id)
            die_ids.add(dag.die_id)
        if {plan_id for dag in self.dags for plan_id in dag.fusion_plan_ids} != set(
            self.fusion_plan_ids
        ):
            raise SchemaError("fusion plans must appear in at least one DAG", path=f"{path}.fusion_plan_ids")
        if {
            plan_id
            for dag in self.dags
            for plan_id in dag.standalone_collective_plan_ids
        } != set(self.standalone_collective_plan_ids):
            raise SchemaError("standalone plans must appear in at least one DAG", path=f"{path}.standalone_collective_plan_ids")
        if {
            transfer_id
            for dag in self.dags
            for transfer_id in dag.state_transfer_ids
        } != transfer_ids:
            raise SchemaError(
                "state transfers must appear in at least one DAG",
                path=f"{path}.state_transfers",
            )
        expected_id = stable_artifact_id(
            "ir2_projection_result",
            self._semantic_key(),
            schema_version=IR2_PROJECTION_RESULT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        fusion_plans: tuple[FusionPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
        path: str = "ir2_projection_result",
    ) -> None:
        self.validate(path)
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("projection references a different IR-1", path=f"{path}.source_ir1_id")
        expected_manifest_id = (
            ir1.persistent_state_manifest.id
            if ir1.persistent_state_manifest is not None
            else None
        )
        if self.source_state_manifest_id != expected_manifest_id:
            raise SchemaError(
                "projection persistent-state manifest provenance disagrees with IR-1",
                path=f"{path}.source_state_manifest_id",
            )
        for index, contract in enumerate(self.state_transfers):
            contract.validate_against(
                ir1, f"{path}.state_transfers[{index}]"
            )
        if self.state_transfers and type(self.state_transfers[0]) in (
            SlicedKvStateTransferContract,
            SegmentedKvStateTransferContract,
        ):
            manifest = ir1.persistent_state_manifest
            assert manifest is not None
            access_index = {
                access.id: access for access in ir1.state_accesses
            }
            declaration_index = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }

            def sliced_transfer_key(
                contract: StateTransferLike,
            ) -> tuple[object, ...]:
                assert isinstance(
                    contract,
                    (SlicedKvStateTransferContract, SegmentedKvStateTransferContract),
                )
                source_access = access_index[
                    contract.source_state_access_ref
                ]
                destination_access = access_index[
                    contract.destination_state_access_ref
                ]
                identity = declaration_index[
                    source_access.state_ref
                ].identity
                return (
                    identity.layer_index,
                    identity.request_ref,
                    source_access.rank,
                    destination_access.rank,
                    contract.source_local_offset,
                    contract.destination_local_offset,
                    identity.kind.value,
                    contract.cross_group_route_ref,
                    contract.id,
                )

            if self.state_transfers != tuple(
                sorted(self.state_transfers, key=sliced_transfer_key)
            ):
                raise SchemaError(
                    "sliced or segmented transfers must follow IR-1 lineage/route/slice order",
                    path=f"{path}.state_transfers",
                )
        ir1_node_index = {node.id: node for node in ir1.nodes}
        has_cross_instance_edge = any(
            ir1_node_index[edge.source_node].instance_id
            != ir1_node_index[edge.destination_node].instance_id
            for edge in ir1.edges
        )
        if has_cross_instance_edge and (
            not self.state_transfers
            or type(self.state_transfers[0]) not in (
                SlicedKvStateTransferContract,
                SegmentedKvStateTransferContract,
            )
        ):
            raise SchemaError(
                "cross-instance IR-1 edges require sliced or segmented state-transfer provenance",
                path=f"{path}.state_transfers",
            )
        fusion_index = {plan.id: plan for plan in fusion_plans}
        standalone_index = {plan.id: plan for plan in standalone_plans}
        expected_fusion_plan_ids = tuple(plan.id for plan in fusion_plans)
        expected_standalone_plan_ids = tuple(plan.id for plan in standalone_plans)
        if (
            len(fusion_index) != len(fusion_plans)
            or self.fusion_plan_ids != expected_fusion_plan_ids
        ):
            raise SchemaError(
                "fusion plan ids must exactly follow the input plan tuple",
                path=f"{path}.fusion_plan_ids",
            )
        if (
            len(standalone_index) != len(standalone_plans)
            or self.standalone_collective_plan_ids
            != expected_standalone_plan_ids
        ):
            raise SchemaError(
                "standalone plan ids must exactly follow the input plan tuple",
                path=f"{path}.standalone_collective_plan_ids",
            )
        for plan in fusion_plans:
            plan.validate_against(ir1)
            chunk_index = {chunk.chunk_id: chunk for chunk in plan.chunk_slices}
            if plan.physical_output_layout != plan.logical_output_layout or any(
                entry.logical_owner_rank != entry.physical_owner_rank
                or entry.physical_owner_rank
                != chunk_index[entry.chunk_id].owner_rank
                for entry in plan.output_permutation
            ):
                raise SchemaError(
                    "MVP executable projection requires identity output ownership and layout",
                    path=f"{path}.fusion_plan_ids",
                )
        for plan in standalone_plans:
            plan.validate_against(ir1)
        if tuple(dag.die_id for dag in self.dags) != tuple(
            die.id for die in ir1.fabric.dies
        ):
            raise SchemaError(
                "DAGs must exactly follow IR-1 fabric die order",
                path=f"{path}.dags",
            )
        for index, dag in enumerate(self.dags):
            dag.validate_against(
                ir1,
                fusion_plans,
                standalone_plans,
                path=f"{path}.dags[{index}]",
            )

        groups = {group.id: group for group in ir1.groups}
        skeletons = {skeleton.id: skeleton for skeleton in ir1.fused_op_skeletons}
        fused_member_ids = tuple(
            member_id
            for plan in fusion_plans
            for member_id in skeletons[plan.fused_op_id].member_node_ids
        )
        standalone_node_ids = tuple(plan.op_id for plan in standalone_plans)
        if (
            len(set(fused_member_ids)) != len(fused_member_ids)
            or len(set(standalone_node_ids)) != len(standalone_node_ids)
            or set(fused_member_ids).intersection(standalone_node_ids)
        ):
            raise SchemaError(
                "fusion and standalone plans must partition covered IR-1 nodes",
                path=f"{path}.dags",
            )
        covered_node_ids = set(fused_member_ids).union(standalone_node_ids)
        ordinary_nodes = tuple(
            node for node in ir1.nodes if node.id not in covered_node_ids
        )
        uncovered_collectives = tuple(
            node.id for node in ordinary_nodes if node.kind is OpKind.COLLECTIVE
        )
        if uncovered_collectives:
            raise SchemaError(
                "collective nodes require a fusion or standalone plan",
                path=f"{path}.dags",
            )

        def plan_die_ids(
            plan: FusionPlan | StandaloneCollectivePlan,
        ) -> set[int]:
            group = groups[plan.group_ref]
            result = {placement.die_id for placement in group.placements}
            for program in plan.rank_programs:
                for action in program.actions:
                    if (
                        action.kind is FusionActionKind.SEND
                        and action.expected_route is not None
                    ):
                        result.update(action.expected_route[1:-1])
            return result

        for dag_index, dag in enumerate(self.dags):
            expected_local_fusion_ids = tuple(
                plan.id for plan in fusion_plans if dag.die_id in plan_die_ids(plan)
            )
            expected_local_standalone_ids = tuple(
                plan.id
                for plan in standalone_plans
                if dag.die_id in plan_die_ids(plan)
            )
            expected_local_ordinary_ids = tuple(
                node.id
                for node in ordinary_nodes
                if any(
                    placement.die_id == dag.die_id
                    for placement in groups[node.execution_group_ref].placements
                )
            )
            if dag.fusion_plan_ids != expected_local_fusion_ids:
                raise SchemaError(
                    "local fusion plan ids must follow filtered input plan order",
                    path=f"{path}.dags[{dag_index}].fusion_plan_ids",
                )
            if (
                dag.standalone_collective_plan_ids
                != expected_local_standalone_ids
            ):
                raise SchemaError(
                    "local standalone plan ids must follow filtered input plan order",
                    path=(
                        f"{path}.dags[{dag_index}]"
                        ".standalone_collective_plan_ids"
                    ),
                )
            if dag.ordinary_node_ids != expected_local_ordinary_ids:
                raise SchemaError(
                    "local ordinary node ids must exactly cover placements in filtered IR-1 node order",
                    path=f"{path}.dags[{dag_index}].ordinary_node_ids",
                )
        expected_ordinary: dict[tuple[str, int], int] = {}
        for node in ordinary_nodes:
            group = groups[node.execution_group_ref]
            for placement in group.placements:
                expected_ordinary[(node.id, placement.rank)] = placement.die_id
        projected_ordinary: dict[
            tuple[str, int], list[tuple[IntraDieDAG, SemanticTask]]
        ] = {}
        for dag in self.dags:
            for task in dag.tasks:
                origin = task.origin_ref
                if isinstance(origin, OrdinaryNodeOrigin):
                    projected_ordinary.setdefault(
                        (origin.op_id, origin.rank), []
                    ).append((dag, task))
        if set(projected_ordinary) != set(expected_ordinary):
            raise SchemaError(
                "ordinary projection must exactly cover every execution-group rank",
                path=f"{path}.dags",
            )
        for key, expected_die in expected_ordinary.items():
            occurrences = projected_ordinary[key]
            if (
                len(occurrences) != 1
                or occurrences[0][0].die_id != expected_die
                or occurrences[0][1].kind is not SemanticTaskKind.COMP
            ):
                raise SchemaError(
                    "each ordinary node/rank must be projected once on its placed die",
                    path=f"{path}.dags",
                )
        expected: dict[
            tuple[str, str, int, str],
            tuple[FusionPlan | StandaloneCollectivePlan, int, FusionAction],
        ] = {}
        for category, plans in (("fusion", fusion_plans), ("standalone", standalone_plans)):
            for plan in plans:
                group = groups[plan.group_ref]
                rank_to_die = {placement.rank: placement.die_id for placement in group.placements}
                for program in plan.rank_programs:
                    for action in program.actions:
                        expected[(category, plan.id, program.rank, action.id)] = (
                            plan,
                            rank_to_die[program.rank],
                            action,
                        )

        projected: dict[tuple[str, str, int, str], list[tuple[IntraDieDAG, SemanticTask]]] = {}
        transits: dict[tuple[str, str, int, str], list[tuple[IntraDieDAG, SemanticTask]]] = {}
        for dag in self.dags:
            for task in dag.tasks:
                origin = task.origin_ref
                if isinstance(origin, FusedNodeOrigin):
                    key = ("fusion", origin.plan_id, origin.rank, origin.action_id)
                elif isinstance(origin, StandaloneNodeOrigin):
                    key = (
                        "standalone",
                        origin.collective_plan_id,
                        origin.rank,
                        origin.action_id,
                    )
                else:
                    continue
                target = transits if task.kind is SemanticTaskKind.TRANSIT else projected
                target.setdefault(key, []).append((dag, task))
        if set(projected) != set(expected):
            raise SchemaError("projected action set does not exactly match plan actions", path=f"{path}.dags")
        for key, (plan, expected_die, action) in expected.items():
            occurrences = projected[key]
            if len(occurrences) != 1:
                raise SchemaError("each plan action must be projected exactly once", path=f"{path}.dags")
            dag, task = occurrences[0]
            if dag.die_id != expected_die or task.kind.value != action.kind.value:
                raise SchemaError("action rank/kind projection is not exact", path=f"{path}.dags")

        ir1_values = {value.id: value for value in ir1.values}
        for plan in fusion_plans:
            value_id = plan.chunk_slices[0].value_id
            reduce_slices = tuple(
                projected[("fusion", plan.id, program.rank, action.id)][0][1].tensor_slice
                for program in plan.rank_programs
                for action in program.actions
                if action.kind is FusionActionKind.REDUCE
            )
            if any(tensor_slice is None for tensor_slice in reduce_slices):
                raise SchemaError(
                    "projected REDUCE writers require explicit tensor slices",
                    path=f"{path}.dags",
                )
            _validate_rectangular_slices(
                tuple(tensor_slice for tensor_slice in reduce_slices if tensor_slice is not None),
                value_id=value_id,
                value_shape=ir1_values[value_id].shape,
                require_full_cover=True,
                path=f"{path}.dags",
            )

        for key, occurrences in transits.items():
            source = expected.get(key)
            if source is None or source[2].kind is not FusionActionKind.SEND:
                raise SchemaError(
                    "TRANSIT must derive from a projected SEND",
                    path=f"{path}.dags",
                )

        node_order = {node.id: index for index, node in enumerate(ir1.nodes)}
        state_access_order = {
            access.id: index
            for index, access in enumerate(ir1.state_accesses)
        }
        fusion_action_order: dict[
            tuple[str, int, str], tuple[int, int, int]
        ] = {}
        for plan in fusion_plans:
            anchor_id = skeletons[plan.fused_op_id].member_node_ids[0]
            for program_index, program in enumerate(plan.rank_programs):
                for action_index, action in enumerate(program.actions):
                    fusion_action_order[(plan.id, program.rank, action.id)] = (
                        node_order[anchor_id],
                        program_index,
                        action_index,
                    )
        standalone_action_order: dict[
            tuple[str, int, str], tuple[int, int, int]
        ] = {}
        for plan in standalone_plans:
            for program_index, program in enumerate(plan.rank_programs):
                for action_index, action in enumerate(program.actions):
                    standalone_action_order[
                        (plan.id, program.rank, action.id)
                    ] = (
                        node_order[plan.op_id],
                        program_index,
                        action_index,
                    )

        state_access_index = {access.id: access for access in ir1.state_accesses}
        route_index = {
            route.id: route
            for group in ir1.groups
            for route in group.embedding.routes
        }
        cross_route_index = {route.id: route for route in ir1.cross_routes}

        def transfer_route(
            contract: StateTransferLike,
        ) -> PairRoute | CrossGroupRoute:
            if isinstance(
                contract,
                (SlicedKvStateTransferContract, SegmentedKvStateTransferContract),
            ):
                return cross_route_index[contract.cross_group_route_ref]
            return route_index[contract.pair_route_ref]
        transfer_task_order: dict[
            str, tuple[int, int, int, int, int, int]
        ] = {}
        for transfer_ordinal, contract in enumerate(self.state_transfers):
            source_access = state_access_index[
                contract.source_state_access_ref
            ]
            destination_access = state_access_index[
                contract.destination_state_access_ref
            ]
            route = transfer_route(contract)
            segment_indices = (
                tuple(range(len(contract.segments)))
                if isinstance(contract, SegmentedKvStateTransferContract)
                else (None,)
            )
            for segment_order, segment_index in enumerate(segment_indices):
                transfer_task_order[canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.SEND,
                    route.die_path[0], segment_index,
                )] = (
                    node_order[source_access.node_ref],
                    source_access.rank,
                    2,
                    0,
                    transfer_ordinal,
                    segment_order,
                )
                transfer_task_order[canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.RECV,
                    route.die_path[-1], segment_index,
                )] = (
                    node_order[destination_access.node_ref],
                    destination_access.rank,
                    0,
                    -1,
                    transfer_ordinal,
                    segment_order * 2,
                )
                transfer_task_order[canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.WAIT,
                    route.die_path[-1], segment_index,
                )] = (
                    node_order[destination_access.node_ref],
                    destination_access.rank,
                    0,
                    -1,
                    transfer_ordinal,
                    segment_order * 2 + 1,
                )
                for hop_index, die_id in enumerate(
                    route.die_path[1:-1], 1
                ):
                    transfer_task_order[canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.TRANSIT, die_id,
                        segment_index,
                    )] = (
                        len(ir1.nodes),
                        route.source_rank,
                        2,
                        0,
                        transfer_ordinal,
                        (
                            hop_index
                            if segment_index is None
                            else segment_order
                        ),
                    )

        def canonical_task_key(
            task: SemanticTask,
            task_path: str,
        ) -> tuple[int, int, int, int, int, int]:
            origin = task.origin_ref
            if isinstance(origin, StateIoOrigin):
                if (
                    origin.node_ref not in node_order
                    or origin.state_access_ref not in state_access_order
                ):
                    raise SchemaError(
                        "state task has no IR-1 source/access-order key",
                        path=f"{task_path}.origin_ref",
                    )
                return (
                    node_order[origin.node_ref],
                    origin.rank,
                    0 if task.kind is SemanticTaskKind.DMA_IN else 3,
                    0,
                    state_access_order[origin.state_access_ref],
                    0,
                )
            if isinstance(origin, StateTransferOrigin):
                expected_key = transfer_task_order.get(task.id)
                if expected_key is None:
                    raise SchemaError(
                        "state transfer task has no contract-derived order key",
                        path=f"{task_path}.origin_ref",
                    )
                return expected_key
            if isinstance(origin, OrdinaryNodeOrigin):
                if origin.op_id not in node_order:
                    raise SchemaError(
                        "ordinary task has no IR-1 source-order key",
                        path=f"{task_path}.origin_ref.op_id",
                    )
                return (node_order[origin.op_id], origin.rank, 1, 0, 0, 0)
            if isinstance(origin, FusedNodeOrigin):
                base = fusion_action_order.get(
                    (origin.plan_id, origin.rank, origin.action_id)
                )
            else:
                base = standalone_action_order.get(
                    (
                        origin.collective_plan_id,
                        origin.rank,
                        origin.action_id,
                    )
                )
            if base is None:
                raise SchemaError(
                    "planned task has no input-plan tuple key",
                    path=f"{task_path}.origin_ref",
                )
            return (
                base[0],
                base[1],
                1,
                base[2],
                0,
                1 if task.kind is SemanticTaskKind.TRANSIT else 0,
            )
        def canonical_region_key(
            region: IntraDieRegion,
            task_index: dict[str, SemanticTask],
            region_path: str,
        ) -> tuple[int, int, int, int, int, int]:
            if not region.task_ids:
                raise SchemaError(
                    "region has no canonical task-order key",
                    path=region_path,
                )
            return min(
                canonical_task_key(
                    task_index[task_id],
                    f"{region_path}.task_ids",
                )
                for task_id in region.task_ids
            )



        for dag_index, dag in enumerate(self.dags):
            task_keys = tuple(
                canonical_task_key(task, f"{path}.dags[{dag_index}].tasks[{index}]")
                for index, task in enumerate(dag.tasks)
            )
            if len(set(task_keys)) != len(task_keys) or task_keys != tuple(
                sorted(task_keys)
            ):
                raise SchemaError(
                    "tasks must follow canonical unit/rank-program/action order",
                    path=f"{path}.dags[{dag_index}].tasks",
                )

            expected_value_ids = tuple(
                dict.fromkeys(
                    value_id
                    for task in dag.tasks
                    for value_id in task.read_values + task.write_values
                    if value_id
                    not in {value.id for value in dag.state_staging_values}
                )
            )
            if tuple(value.id for value in dag.values) != expected_value_ids:
                raise SchemaError(
                    "values must follow canonical task first-use order",
                    path=f"{path}.dags[{dag_index}].values",
                )

            task_index = {task.id: task for task in dag.tasks}
            flow_keys: list[tuple[int, int, int, int, int, int]] = []
            for flow_index, flow in enumerate(dag.flows):
                if len(flow.task_ids) != 1:
                    raise SchemaError(
                        "each local flow replica, including TRANSIT, must name one canonical task",
                        path=(
                            f"{path}.dags[{dag_index}].flows[{flow_index}]"
                            ".task_ids"
                        ),
                    )
                flow_keys.append(
                    canonical_task_key(
                        task_index[flow.task_ids[0]],
                        f"{path}.dags[{dag_index}].flows[{flow_index}]",
                    )
                )
            if len(set(flow_keys)) != len(flow_keys) or tuple(
                flow_keys
            ) != tuple(sorted(flow_keys)):
                raise SchemaError(
                    "flows must follow their unique canonical task order",
                    path=f"{path}.dags[{dag_index}].flows",
                )

            region_keys: list[tuple[int, int, int, int, int, int]] = []
            for region_index, region in enumerate(dag.regions):
                region_path = (
                    f"{path}.dags[{dag_index}]"
                    f".regions[{region_index}]"
                )
                region_keys.append(
                    canonical_region_key(
                        region,
                        task_index,
                        region_path,
                    )
                )
            if len(set(region_keys)) != len(region_keys) or tuple(
                region_keys
            ) != tuple(sorted(region_keys)):
                raise SchemaError(
                    "regions must follow canonical unit/ordinary-rank order",
                    path=f"{path}.dags[{dag_index}].regions",
                )

        dag_by_die = {dag.die_id: dag for dag in self.dags}
        expected_flow_replicas: set[tuple[int, str]] = set()
        for key, (plan, _expected_die, action) in expected.items():
            if action.kind is not FusionActionKind.SEND:
                if key in transits:
                    raise SchemaError(
                        "only SEND may own TRANSIT tasks", path=f"{path}.dags"
                    )
                continue
            assert action.logical_channel is not None
            source_dag, source_task = projected[key][0]
            source_origin = source_task.origin_ref
            assert isinstance(source_origin, (FusedNodeOrigin, StandaloneNodeOrigin))
            flow_id = canonical_semantic_flow_id(
                source_origin, action.logical_channel
            )
            receiver_key = next(
                (
                    candidate_key
                    for candidate_key, (_candidate_plan, _die, candidate_action) in expected.items()
                    if candidate_key[0:2] == key[0:2]
                    and candidate_key[2] == action.peer_rank
                    and candidate_action.kind is FusionActionKind.RECV
                    and candidate_action.peer_rank == key[2]
                    and candidate_action.logical_channel == action.logical_channel
                ),
                None,
            )
            if receiver_key is None:
                raise SchemaError(
                    "SEND has no exact paired RECV", path=f"{path}.dags"
                )
            destination_dag, destination_task = projected[receiver_key][0]
            chunk = next(
                item for item in plan.chunk_slices if item.id == action.slice_ref
            )
            expected_slice = TensorSlice(
                chunk.value_id, chunk.offset, chunk.shape
            )
            group = groups[plan.group_ref]
            route = next(
                item
                for item in group.embedding.routes
                if item.source_rank == key[2]
                and item.destination_rank == action.peer_rank
                and item.die_path == action.expected_route
            )
            expected_transit_dies = tuple(action.expected_route[1:-1])
            actual_transits = transits.get(key, ())
            actual_transit_dies = tuple(
                dag.die_id for dag, _task in actual_transits
            )
            if (
                len(set(actual_transit_dies)) != len(actual_transit_dies)
                or set(actual_transit_dies) != set(expected_transit_dies)
            ):
                raise SchemaError(
                    "selected route requires one canonical TRANSIT on every intermediate die",
                    path=f"{path}.dags",
                )
            transit_by_die = {
                dag.die_id: task for dag, task in actual_transits
            }
            for hop_index, die_id in enumerate(action.expected_route):
                dag = dag_by_die[die_id]
                expected_flow_replicas.add((die_id, flow_id))
                local_task = (
                    source_task
                    if hop_index == 0
                    else destination_task
                    if hop_index == len(action.expected_route) - 1
                    else transit_by_die[die_id]
                )
                flow_matches = tuple(
                    flow for flow in dag.flows if flow.id == flow_id
                )
                if len(flow_matches) != 1:
                    raise SchemaError(
                        "route die requires exactly one canonical flow replica",
                        path=f"{path}.dags",
                    )
                flow = flow_matches[0]
                if (
                    flow.logical_channel,
                    flow.pair_route_ref,
                    flow.source_rank,
                    flow.destination_rank,
                    flow.source_die,
                    flow.destination_die,
                    flow.die_path,
                    flow.tensor_slice,
                    flow.bytes,
                    flow.dtype,
                    flow.task_ids,
                ) != (
                    action.logical_channel,
                    route.id,
                    key[2],
                    action.peer_rank,
                    source_dag.die_id,
                    destination_dag.die_id,
                    action.expected_route,
                    expected_slice,
                    action.bytes,
                    action.dtype,
                    (local_task.id,),
                ):
                    raise SchemaError(
                        "semantic flow replica is not exact",
                        path=f"{path}.dags",
                    )
                if local_task.flow_id != flow_id:
                    raise SchemaError(
                        "transport task must use canonical flow identity",
                        path=f"{path}.dags",
                    )
                if hop_index in (0, len(action.expected_route) - 1):
                    continue
                region = next(
                    (
                        region
                        for region in dag.regions
                        if (
                            region.fusion_plan_id == plan.id
                            if key[0] == "fusion"
                            else region.standalone_collective_plan_id == plan.id
                        )
                    ),
                    None,
                )
                if region is None:
                    raise SchemaError(
                        "TRANSIT die must inherit the source plan region",
                        path=f"{path}.dags",
                    )
                expected_sync = SyncContract(
                    canonical_transit_completion_event(flow_id, die_id),
                    None,
                    None,
                )
                if (
                    local_task.origin_ref,
                    local_task.region_id,
                    local_task.op_kind,
                    local_task.member_id,
                    local_task.flow_id,
                    local_task.chunk_id,
                    local_task.collective_step,
                    local_task.source_rank,
                    local_task.destination_rank,
                    local_task.tensor_slice,
                    local_task.bytes,
                    local_task.dtype,
                    local_task.shape,
                    local_task.read_values,
                    local_task.write_values,
                    local_task.compute,
                    local_task.reduction,
                    local_task.sync,
                    local_task.deps,
                ) != (
                    source_origin,
                    region.id,
                    OpKind.COLLECTIVE,
                    action.member_id,
                    flow_id,
                    action.chunk_id,
                    action.collective_step,
                    key[2],
                    action.peer_rank,
                    expected_slice,
                    action.bytes,
                    action.dtype,
                    chunk.shape,
                    (),
                    (),
                    None,
                    None,
                    expected_sync,
                    (),
                ):
                    raise SchemaError(
                        "TRANSIT fields, source origin, region, hop payload, sync and local deps must be exact",
                        path=f"{path}.dags",
                    )
        # State-transfer flows form an independent exact quotient.  Their
        # route-global logical payload id intentionally differs from each
        # endpoint-local staging id; no collective-flow rule is weakened.
        access_index = {access.id: access for access in ir1.state_accesses}
        manifest = ir1.persistent_state_manifest
        if self.state_transfers and manifest is None:
            raise SchemaError(
                "state transfers require persistent-state provenance",
                path=f"{path}.state_transfers",
            )
        declaration_index = (
            {declaration.id: declaration for declaration in manifest.declarations}
            if manifest is not None
            else {}
        )
        actual_transfer_tasks: dict[
            tuple[str, int | None, int, SemanticTaskKind], list[SemanticTask]
        ] = {}
        for dag in self.dags:
            expected_local_transfer_ids = tuple(
                contract.id
                for contract in self.state_transfers
                if dag.die_id in transfer_route(contract).die_path
            )
            if dag.state_transfer_ids != expected_local_transfer_ids:
                raise SchemaError(
                    "state_transfer_ids must exactly follow filtered contract order",
                    path=f"{path}.dags",
                )
            for task in dag.tasks:
                if isinstance(task.origin_ref, StateTransferOrigin):
                    actual_transfer_tasks.setdefault(
                        (
                            task.origin_ref.state_transfer_ref,
                            task.origin_ref.segment_index,
                            dag.die_id,
                            task.kind,
                        ),
                        [],
                    ).append(task)

        expected_transfer_task_keys: set[
            tuple[str, int | None, int, SemanticTaskKind]
        ] = set()
        for contract in self.state_transfers:
            source_access = access_index[contract.source_state_access_ref]
            destination_access = access_index[
                contract.destination_state_access_ref
            ]
            source_declaration = declaration_index[source_access.state_ref]
            destination_declaration = declaration_index[
                destination_access.state_ref
            ]
            route = transfer_route(contract)
            source_die = route.die_path[0]
            destination_die = route.die_path[-1]
            source_dag = dag_by_die[source_die]
            destination_dag = dag_by_die[destination_die]
            source_staging_id = canonical_state_staging_value_id(
                source_access.id
            )
            destination_staging_id = canonical_state_staging_value_id(
                destination_access.id
            )
            if isinstance(
                contract,
                (SlicedKvStateTransferContract, SegmentedKvStateTransferContract),
            ):
                source_dma_kind = SemanticTaskKind.DMA_OUT
                source_offset = contract.source_local_offset
                source_shape = contract.source_local_shape
                destination_offset = contract.destination_local_offset
                destination_shape = contract.destination_local_shape
                payload_bytes = contract.bytes
            else:
                source_dma_kind = SemanticTaskKind.DMA_IN
                source_offset = (0,) * len(source_declaration.shape)
                source_shape = source_declaration.shape
                destination_offset = (
                    (0,) * len(destination_declaration.shape)
                )
                destination_shape = destination_declaration.shape
                payload_bytes = source_declaration.tensor_bytes

            def state_dma(
                dag: IntraDieDAG,
                access_ref: str,
                kind: SemanticTaskKind,
            ) -> SemanticTask:
                matches = tuple(
                    task
                    for task in dag.tasks
                    if (
                        isinstance(task.origin_ref, StateIoOrigin)
                        and task.origin_ref.state_access_ref == access_ref
                        and task.kind is kind
                    )
                )
                if len(matches) != 1:
                    raise SchemaError(
                        "transfer endpoint requires one exact state DMA",
                        path=f"{path}.dags",
                    )
                return matches[0]

            source_dma = state_dma(
                source_dag,
                source_access.id,
                source_dma_kind,
            )
            destination_dma = state_dma(
                destination_dag,
                destination_access.id,
                SemanticTaskKind.DMA_OUT,
            )
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_targets = source_dma.dma.access_task_refs
            destination_targets = destination_dma.dma.access_task_refs
            expected_source_dma_deps = (
                source_targets
                if source_dma_kind is SemanticTaskKind.DMA_OUT
                else ()
            )
            if (
                source_dma.deps != expected_source_dma_deps
                or destination_dma.deps != destination_targets
            ):
                raise SchemaError(
                    "state transfer endpoint DMA dependencies are not exact",
                    path=f"{path}.dags",
                )

            if isinstance(contract, SegmentedKvStateTransferContract):
                source_region_tasks: list[str] = []
                destination_region_tasks: list[str] = []
                transit_region_tasks: dict[int, list[str]] = {
                    die_id: [] for die_id in route.die_path[1:-1]
                }
                for segment_index, segment in enumerate(contract.segments):
                    flow_id = canonical_state_transfer_flow_id(
                        contract.id, segment_index
                    )
                    logical_slice = TensorSlice(
                        canonical_state_transfer_payload_id(
                            contract.id, segment_index
                        ),
                        (0,) * len(segment.source_local_shape),
                        segment.source_local_shape,
                    )
                    send_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.SEND, source_die,
                        segment_index,
                    )
                    recv_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.RECV, destination_die,
                        segment_index,
                    )
                    wait_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.WAIT, destination_die,
                        segment_index,
                    )
                    recv_event = canonical_state_transfer_completion_event(
                        contract.id, SemanticTaskKind.RECV, destination_die,
                        segment_index,
                    )
                    expected_tasks: dict[
                        tuple[int, SemanticTaskKind], SemanticTask
                    ] = {
                        (source_die, SemanticTaskKind.SEND): SemanticTask(
                            id=send_id,
                            kind=SemanticTaskKind.SEND,
                            origin_ref=StateTransferOrigin(
                                OriginKind.STATE_TRANSFER,
                                contract.id,
                                route.source_rank,
                                segment_index,
                            ),
                            region_id=canonical_state_transfer_region_id(
                                contract.id, source_die
                            ),
                            op_kind=OpKind.P2P,
                            member_id=None,
                            flow_id=flow_id,
                            chunk_id=None,
                            collective_step=None,
                            source_rank=route.source_rank,
                            destination_rank=route.destination_rank,
                            tensor_slice=TensorSlice(
                                source_staging_id,
                                segment.source_local_offset,
                                segment.source_local_shape,
                            ),
                            bytes=segment.bytes,
                            dtype=source_declaration.dtype,
                            shape=segment.source_local_shape,
                            read_values=(source_staging_id,),
                            write_values=(),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(
                                canonical_state_transfer_completion_event(
                                    contract.id, SemanticTaskKind.SEND,
                                    source_die, segment_index,
                                ),
                                None,
                                None,
                            ),
                            deps=source_targets,
                            dma=None,
                        ),
                        (destination_die, SemanticTaskKind.RECV): SemanticTask(
                            id=recv_id,
                            kind=SemanticTaskKind.RECV,
                            origin_ref=StateTransferOrigin(
                                OriginKind.STATE_TRANSFER,
                                contract.id,
                                route.destination_rank,
                                segment_index,
                            ),
                            region_id=canonical_state_transfer_region_id(
                                contract.id, destination_die
                            ),
                            op_kind=OpKind.P2P,
                            member_id=None,
                            flow_id=flow_id,
                            chunk_id=None,
                            collective_step=None,
                            source_rank=route.source_rank,
                            destination_rank=route.destination_rank,
                            tensor_slice=TensorSlice(
                                destination_staging_id,
                                segment.destination_local_offset,
                                segment.destination_local_shape,
                            ),
                            bytes=segment.bytes,
                            dtype=destination_declaration.dtype,
                            shape=segment.destination_local_shape,
                            read_values=(),
                            write_values=(destination_staging_id,),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(recv_event, None, None),
                            deps=(),
                            dma=None,
                        ),
                        (destination_die, SemanticTaskKind.WAIT): SemanticTask(
                            id=wait_id,
                            kind=SemanticTaskKind.WAIT,
                            origin_ref=StateTransferOrigin(
                                OriginKind.STATE_TRANSFER,
                                contract.id,
                                route.destination_rank,
                                segment_index,
                            ),
                            region_id=canonical_state_transfer_region_id(
                                contract.id, destination_die
                            ),
                            op_kind=OpKind.P2P,
                            member_id=None,
                            flow_id=None,
                            chunk_id=None,
                            collective_step=None,
                            source_rank=None,
                            destination_rank=None,
                            tensor_slice=None,
                            bytes=0,
                            dtype=None,
                            shape=(),
                            read_values=(),
                            write_values=(),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(
                                canonical_state_transfer_completion_event(
                                    contract.id, SemanticTaskKind.WAIT,
                                    destination_die, segment_index,
                                ),
                                recv_event,
                                None,
                            ),
                            deps=(recv_id,),
                            dma=None,
                        ),
                    }
                    for die_id in route.die_path[1:-1]:
                        expected_tasks[
                            (die_id, SemanticTaskKind.TRANSIT)
                        ] = SemanticTask(
                            id=canonical_state_transfer_task_id(
                                contract.id, SemanticTaskKind.TRANSIT,
                                die_id, segment_index,
                            ),
                            kind=SemanticTaskKind.TRANSIT,
                            origin_ref=StateTransferOrigin(
                                OriginKind.STATE_TRANSFER,
                                contract.id,
                                route.source_rank,
                                segment_index,
                            ),
                            region_id=canonical_state_transfer_region_id(
                                contract.id, die_id
                            ),
                            op_kind=OpKind.P2P,
                            member_id=None,
                            flow_id=flow_id,
                            chunk_id=None,
                            collective_step=None,
                            source_rank=route.source_rank,
                            destination_rank=route.destination_rank,
                            tensor_slice=logical_slice,
                            bytes=segment.bytes,
                            dtype=source_declaration.dtype,
                            shape=segment.source_local_shape,
                            read_values=(),
                            write_values=(),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(
                                canonical_state_transfer_completion_event(
                                    contract.id, SemanticTaskKind.TRANSIT,
                                    die_id, segment_index,
                                ),
                                None,
                                None,
                            ),
                            deps=(),
                            dma=None,
                        )

                    for (die_id, kind), expected_task in expected_tasks.items():
                        task_key = (
                            contract.id, segment_index, die_id, kind
                        )
                        expected_transfer_task_keys.add(task_key)
                        matches = actual_transfer_tasks.get(task_key, [])
                        if len(matches) != 1 or matches[0] != expected_task:
                            raise SchemaError(
                                "segmented state transfer task fields are not contract-exact",
                                path=f"{path}.dags",
                            )

                    for dag in self.dags:
                        for task in dag.tasks:
                            has_wait = wait_id in task.deps
                            should_wait = (
                                dag.die_id == destination_die
                                and task.id in destination_targets
                            )
                            if has_wait != should_wait:
                                raise SchemaError(
                                    "every segmented destination WAIT must feed exactly every destination access target",
                                    path=f"{path}.dags",
                                )

                    local_transport = {
                        source_die: expected_tasks[
                            (source_die, SemanticTaskKind.SEND)
                        ],
                        destination_die: expected_tasks[
                            (destination_die, SemanticTaskKind.RECV)
                        ],
                    }
                    for die_id in route.die_path[1:-1]:
                        local_transport[die_id] = expected_tasks[
                            (die_id, SemanticTaskKind.TRANSIT)
                        ]
                    for die_id in route.die_path:
                        expected_flow_replicas.add((die_id, flow_id))
                        dag = dag_by_die[die_id]
                        local_task = local_transport[die_id]
                        flow_matches = tuple(
                            flow for flow in dag.flows
                            if flow.id == flow_id
                        )
                        expected_flow = SemanticFlow(
                            id=flow_id,
                            logical_channel=(
                                f"state_transfer.{contract.id}.segment.{segment_index}"
                            ),
                            pair_route_ref=route.id,
                            source_rank=route.source_rank,
                            destination_rank=route.destination_rank,
                            source_die=source_die,
                            destination_die=destination_die,
                            die_path=route.die_path,
                            tensor_slice=logical_slice,
                            bytes=segment.bytes,
                            dtype=source_declaration.dtype,
                            task_ids=(local_task.id,),
                        )
                        if (
                            len(flow_matches) != 1
                            or flow_matches[0] != expected_flow
                        ):
                            raise SchemaError(
                                "segmented state transfer flow replica is not contract-exact",
                                path=f"{path}.dags",
                            )
                    source_region_tasks.append(send_id)
                    destination_region_tasks.extend((recv_id, wait_id))
                    for die_id in route.die_path[1:-1]:
                        transit_region_tasks[die_id].append(
                            expected_tasks[
                                (die_id, SemanticTaskKind.TRANSIT)
                            ].id
                        )

                expected_region_tasks = {
                    source_die: tuple(source_region_tasks),
                    destination_die: tuple(destination_region_tasks),
                    **{
                        die_id: tuple(task_ids)
                        for die_id, task_ids in transit_region_tasks.items()
                    },
                }
                for die_id in route.die_path:
                    expected_region = IntraDieRegion(
                        id=canonical_state_transfer_region_id(
                            contract.id, die_id
                        ),
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.STRICT_STATE_TRANSFER,
                        task_ids=expected_region_tasks[die_id],
                        state_transfer_ref=contract.id,
                    )
                    region_matches = tuple(
                        region for region in dag_by_die[die_id].regions
                        if region.state_transfer_ref == contract.id
                    )
                    if (
                        len(region_matches) != 1
                        or region_matches[0] != expected_region
                    ):
                        raise SchemaError(
                            "segmented state transfer region is not contract-exact",
                            path=f"{path}.dags",
                        )
                continue
            flow_id = canonical_state_transfer_flow_id(contract.id)
            logical_slice = TensorSlice(
                canonical_state_transfer_payload_id(contract.id),
                (0,) * len(source_shape),
                source_shape,
            )
            send_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.SEND, source_die
            )
            recv_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.RECV, destination_die
            )
            wait_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.WAIT, destination_die
            )
            recv_event = canonical_state_transfer_completion_event(
                contract.id, SemanticTaskKind.RECV, destination_die
            )
            expected_tasks: dict[
                tuple[int, SemanticTaskKind], SemanticTask
            ] = {
                (source_die, SemanticTaskKind.SEND): SemanticTask(
                    id=send_id,
                    kind=SemanticTaskKind.SEND,
                    origin_ref=StateTransferOrigin(
                        OriginKind.STATE_TRANSFER,
                        contract.id,
                        route.source_rank,
                    ),
                    region_id=canonical_state_transfer_region_id(
                        contract.id, source_die
                    ),
                    op_kind=OpKind.P2P,
                    member_id=None,
                    flow_id=flow_id,
                    chunk_id=None,
                    collective_step=None,
                    source_rank=route.source_rank,
                    destination_rank=route.destination_rank,
                    tensor_slice=TensorSlice(
                        source_staging_id,
                        source_offset,
                        source_shape,
                    ),
                    bytes=payload_bytes,
                    dtype=source_declaration.dtype,
                    shape=source_shape,
                    read_values=(source_staging_id,),
                    write_values=(),
                    compute=None,
                    reduction=None,
                    sync=SyncContract(
                        canonical_state_transfer_completion_event(
                            contract.id, SemanticTaskKind.SEND, source_die
                        ),
                        None,
                        None,
                    ),
                    deps=source_targets,
                    dma=None,
                ),
                (destination_die, SemanticTaskKind.RECV): SemanticTask(
                    id=recv_id,
                    kind=SemanticTaskKind.RECV,
                    origin_ref=StateTransferOrigin(
                        OriginKind.STATE_TRANSFER,
                        contract.id,
                        route.destination_rank,
                    ),
                    region_id=canonical_state_transfer_region_id(
                        contract.id, destination_die
                    ),
                    op_kind=OpKind.P2P,
                    member_id=None,
                    flow_id=flow_id,
                    chunk_id=None,
                    collective_step=None,
                    source_rank=route.source_rank,
                    destination_rank=route.destination_rank,
                    tensor_slice=TensorSlice(
                        destination_staging_id,
                        destination_offset,
                        destination_shape,
                    ),
                    bytes=payload_bytes,
                    dtype=destination_declaration.dtype,
                    shape=destination_shape,
                    read_values=(),
                    write_values=(destination_staging_id,),
                    compute=None,
                    reduction=None,
                    sync=SyncContract(recv_event, None, None),
                    deps=(),
                    dma=None,
                ),
                (destination_die, SemanticTaskKind.WAIT): SemanticTask(
                    id=wait_id,
                    kind=SemanticTaskKind.WAIT,
                    origin_ref=StateTransferOrigin(
                        OriginKind.STATE_TRANSFER,
                        contract.id,
                        route.destination_rank,
                    ),
                    region_id=canonical_state_transfer_region_id(
                        contract.id, destination_die
                    ),
                    op_kind=OpKind.P2P,
                    member_id=None,
                    flow_id=None,
                    chunk_id=None,
                    collective_step=None,
                    source_rank=None,
                    destination_rank=None,
                    tensor_slice=None,
                    bytes=0,
                    dtype=None,
                    shape=(),
                    read_values=(),
                    write_values=(),
                    compute=None,
                    reduction=None,
                    sync=SyncContract(
                        canonical_state_transfer_completion_event(
                            contract.id, SemanticTaskKind.WAIT, destination_die
                        ),
                        recv_event,
                        None,
                    ),
                    deps=(recv_id,),
                    dma=None,
                ),
            }
            for hop_index, die_id in enumerate(route.die_path[1:-1], 1):
                expected_tasks[(die_id, SemanticTaskKind.TRANSIT)] = SemanticTask(
                    id=canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.TRANSIT, die_id
                    ),
                    kind=SemanticTaskKind.TRANSIT,
                    origin_ref=StateTransferOrigin(
                        OriginKind.STATE_TRANSFER,
                        contract.id,
                        route.source_rank,
                    ),
                    region_id=canonical_state_transfer_region_id(
                        contract.id, die_id
                    ),
                    op_kind=OpKind.P2P,
                    member_id=None,
                    flow_id=flow_id,
                    chunk_id=None,
                    collective_step=None,
                    source_rank=route.source_rank,
                    destination_rank=route.destination_rank,
                    tensor_slice=logical_slice,
                    bytes=payload_bytes,
                    dtype=source_declaration.dtype,
                    shape=source_shape,
                    read_values=(),
                    write_values=(),
                    compute=None,
                    reduction=None,
                    sync=SyncContract(
                        canonical_state_transfer_completion_event(
                            contract.id,
                            SemanticTaskKind.TRANSIT,
                            die_id,
                        ),
                        None,
                        None,
                    ),
                    deps=(),
                    dma=None,
                )

            for (die_id, kind), expected_task in expected_tasks.items():
                task_key = (contract.id, None, die_id, kind)
                expected_transfer_task_keys.add(task_key)
                matches = actual_transfer_tasks.get(task_key, [])
                if len(matches) != 1 or matches[0] != expected_task:
                    raise SchemaError(
                        "state transfer task fields are not contract-exact",
                        path=f"{path}.dags",
                    )

            for dag in self.dags:
                for task in dag.tasks:
                    has_wait = wait_id in task.deps
                    should_wait = (
                        dag.die_id == destination_die
                        and task.id in destination_targets
                    )
                    if has_wait != should_wait:
                        raise SchemaError(
                            "destination WAIT must feed exactly every destination access target",
                            path=f"{path}.dags",
                        )

            local_transport = {
                source_die: expected_tasks[
                    (source_die, SemanticTaskKind.SEND)
                ],
                destination_die: expected_tasks[
                    (destination_die, SemanticTaskKind.RECV)
                ],
            }
            for die_id in route.die_path[1:-1]:
                local_transport[die_id] = expected_tasks[
                    (die_id, SemanticTaskKind.TRANSIT)
                ]
            for die_id in route.die_path:
                expected_flow_replicas.add((die_id, flow_id))
                dag = dag_by_die[die_id]
                local_task = local_transport[die_id]
                flow_matches = tuple(
                    flow for flow in dag.flows if flow.id == flow_id
                )
                expected_flow = SemanticFlow(
                    id=flow_id,
                    logical_channel=f"state_transfer.{contract.id}",
                    pair_route_ref=route.id,
                    source_rank=route.source_rank,
                    destination_rank=route.destination_rank,
                    source_die=source_die,
                    destination_die=destination_die,
                    die_path=route.die_path,
                    tensor_slice=logical_slice,
                    bytes=payload_bytes,
                    dtype=source_declaration.dtype,
                    task_ids=(local_task.id,),
                )
                if len(flow_matches) != 1 or flow_matches[0] != expected_flow:
                    raise SchemaError(
                        "state transfer flow replica is not contract-exact",
                        path=f"{path}.dags",
                    )
                expected_region_tasks = (
                    (recv_id, wait_id)
                    if die_id == destination_die
                    else (local_task.id,)
                )
                expected_region = IntraDieRegion(
                    id=canonical_state_transfer_region_id(
                        contract.id, die_id
                    ),
                    fusion_plan_id=None,
                    standalone_collective_plan_id=None,
                    lowering=RegionLowering.STRICT_STATE_TRANSFER,
                    task_ids=expected_region_tasks,
                    state_transfer_ref=contract.id,
                )
                region_matches = tuple(
                    region
                    for region in dag.regions
                    if region.state_transfer_ref == contract.id
                )
                if (
                    len(region_matches) != 1
                    or region_matches[0] != expected_region
                ):
                    raise SchemaError(
                        "state transfer region is not contract-exact",
                        path=f"{path}.dags",
                    )

        if set(actual_transfer_tasks) != expected_transfer_task_keys:
            raise SchemaError(
                "state transfer tasks must exactly cover every contract route",
                path=f"{path}.dags",
            )

        actual_flow_replicas = {
            (dag.die_id, flow.id) for dag in self.dags for flow in dag.flows
        }
        if actual_flow_replicas != expected_flow_replicas:
            raise SchemaError(
                "semantic flow replicas must exactly cover every selected route die",
                path=f"{path}.dags",
            )


class MemoryRegion(str, Enum):
    SRAM = "sram"
    HBM = "hbm"


class BufferOwnership(str, Enum):
    OWNED = "owned"
    BORROWED = "borrowed"
    ALIASED = "aliased"


class BufferAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class StateUseAccess(str, Enum):
    """Access direction for one task's independent HBM state endpoint."""

    READ = "read"
    WRITE = "write"


class BufferUseRole(str, Enum):
    COMP_INPUT = "comp_input"
    COMP_OUTPUT = "comp_output"
    SEND_SOURCE = "send_source"
    RECV_DESTINATION = "recv_destination"
    REDUCE_INPUT = "reduce_input"
    REDUCE_OUTPUT = "reduce_output"
    LOCAL_COPY_SOURCE = "local_copy_source"
    LOCAL_COPY_DESTINATION = "local_copy_destination"
    DMA_SOURCE = "dma_source"
    DMA_DESTINATION = "dma_destination"


_BUFFER_USE_ROLE_ORDER = {
    role: index
    for index, role in enumerate(
        (
            BufferUseRole.COMP_INPUT,
            BufferUseRole.COMP_OUTPUT,
            BufferUseRole.SEND_SOURCE,
            BufferUseRole.RECV_DESTINATION,
            BufferUseRole.REDUCE_INPUT,
            BufferUseRole.REDUCE_OUTPUT,
            BufferUseRole.LOCAL_COPY_SOURCE,
            BufferUseRole.LOCAL_COPY_DESTINATION,
            BufferUseRole.DMA_SOURCE,
            BufferUseRole.DMA_DESTINATION,
        )
    )
}


@dataclass(frozen=True, slots=True)
class TaskPlacement:
    task_id: str
    core_id: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_id, f"{path}.task_id")
        validate_uint64(self.core_id, f"{path}.core_id")


@dataclass(frozen=True, slots=True)
class BufferBinding:
    id: str
    value_id: str
    tensor_slice: TensorSlice
    core_id: int
    region_ref: str
    region_offset_bytes: int
    size_bytes: int
    alignment_bytes: int
    banks: tuple[int, ...]
    storage_id: str
    alias_of: str | None
    ownership: BufferOwnership
    lifetime_start: int
    lifetime_end_exclusive: int
    dtype: DType
    layout: str

    def validate(self, path: str) -> None:
        for field_name in ("id", "value_id", "region_ref", "storage_id", "layout"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.tensor_slice.value_id != self.value_id:
            raise SchemaError("tensor_slice references a different value", path=f"{path}.tensor_slice.value_id")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        for field_name in (
            "core_id",
            "region_offset_bytes",
            "size_bytes",
            "alignment_bytes",
            "lifetime_start",
            "lifetime_end_exclusive",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.size_bytes == 0 or self.alignment_bytes == 0:
            raise SchemaError("size and alignment must be greater than zero", path=path)
        if self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError("must be a power of two", path=f"{path}.alignment_bytes")
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError(
                "half-open lifetime must be non-empty",
                path=f"{path}.lifetime_end_exclusive",
            )
        if len(set(self.banks)) != len(self.banks):
            raise SchemaError("contains duplicate banks", path=f"{path}.banks")
        if not self.banks:
            raise SchemaError("SRAM binding requires at least one bank", path=f"{path}.banks")
        for index, bank in enumerate(self.banks):
            validate_uint64(bank, f"{path}.banks[{index}]")
        if self.alias_of is not None:
            validate_nonempty(self.alias_of, f"{path}.alias_of")
        if self.ownership is BufferOwnership.ALIASED:
            if self.alias_of is None:
                raise SchemaError("ALIASED binding requires alias_of", path=f"{path}.alias_of")
        elif self.alias_of is not None:
            raise SchemaError(
                "non-ALIASED binding cannot carry alias_of",
                path=f"{path}.alias_of",
            )


@dataclass(frozen=True, slots=True)
class TaskBufferUse:
    task_id: str
    binding_id: str
    access: BufferAccess
    role: BufferUseRole
    operand_index: int
    contribution_rank: int | None
    tensor_slice: TensorSlice

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_id, f"{path}.task_id")
        validate_nonempty(self.binding_id, f"{path}.binding_id")
        if type(self.access) is not BufferAccess:
            raise SchemaError("must be a BufferAccess", path=f"{path}.access")
        if type(self.role) is not BufferUseRole:
            raise SchemaError("must be a BufferUseRole", path=f"{path}.role")
        validate_uint64(self.operand_index, f"{path}.operand_index")
        if type(self.tensor_slice) is not TensorSlice:
            raise SchemaError(
                "must be a TensorSlice",
                path=f"{path}.tensor_slice",
            )
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.contribution_rank is not None:
            validate_uint64(self.contribution_rank, f"{path}.contribution_rank")
        if self.role is BufferUseRole.REDUCE_INPUT:
            if self.contribution_rank is None:
                raise SchemaError(
                    "REDUCE_INPUT requires contribution_rank",
                    path=f"{path}.contribution_rank",
                )
        elif self.contribution_rank is not None:
            raise SchemaError(
                "only REDUCE_INPUT may carry contribution_rank",
                path=f"{path}.contribution_rank",
            )


@dataclass(frozen=True, slots=True)
class TaskStateUse:
    """Bind one scheduled task to one HBM binding owned by the IR-1 manifest."""

    task_id: str
    hbm_binding_ref: str
    access: StateUseAccess

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_id, f"{path}.task_id")
        validate_nonempty(self.hbm_binding_ref, f"{path}.hbm_binding_ref")
        if type(self.access) is not StateUseAccess:
            raise SchemaError(
                "must be a StateUseAccess",
                path=f"{path}.access",
            )


class FlowRouteRole(str, Enum):
    SOURCE = "source"
    TRANSIT = "transit"
    DESTINATION = "destination"


@dataclass(frozen=True, slots=True)
class PortLeg:
    link_ref: str
    port_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.link_ref, f"{path}.link_ref")
        validate_nonempty(self.port_ref, f"{path}.port_ref")


@dataclass(frozen=True, slots=True)
class FlowRouteBinding:
    flow_id: str
    pair_route_ref: str
    role: FlowRouteRole
    ingress: PortLeg | None
    egress: PortLeg | None
    local_noc_path: tuple[tuple[int, int], ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.flow_id, f"{path}.flow_id")
        validate_nonempty(self.pair_route_ref, f"{path}.pair_route_ref")
        if self.ingress is not None:
            self.ingress.validate(f"{path}.ingress")
        if self.egress is not None:
            self.egress.validate(f"{path}.egress")
        expected_legs = {
            FlowRouteRole.SOURCE: (False, True),
            FlowRouteRole.TRANSIT: (True, True),
            FlowRouteRole.DESTINATION: (True, False),
        }[self.role]
        if (self.ingress is not None, self.egress is not None) != expected_legs:
            raise SchemaError("ingress/egress legs disagree with route role", path=path)
        if not self.local_noc_path:
            raise SchemaError("must contain at least one NoC coordinate", path=f"{path}.local_noc_path")
        for coord_index, coordinate in enumerate(self.local_noc_path):
            for axis, value in enumerate(coordinate):
                validate_uint64(value, f"{path}.local_noc_path[{coord_index}][{axis}]")


def _backend_xy_noc_path(
    source: tuple[int, int], destination: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    x, y = source
    destination_x, destination_y = destination
    result = [(x, y)]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        result.append((x, y))
    while y != destination_y:
        y += 1 if destination_y > y else -1
        result.append((x, y))
    return tuple(result)


def _global_route_index(
    ir1: IR1, path: str
) -> dict[str, PairRoute | CrossGroupRoute]:
    result: dict[str, PairRoute | CrossGroupRoute] = {}
    for group_index, group in enumerate(ir1.groups):
        for route_index, route in enumerate(group.embedding.routes):
            if route.id in result:
                raise SchemaError(
                    "PairRoute ids must be globally unique in IR-1",
                    path=f"{path}.groups[{group_index}].embedding.routes[{route_index}].id",
                )
            result[route.id] = route
    for route_index, route in enumerate(ir1.cross_routes):
        if route.id in result:
            raise SchemaError(
                "PairRoute and CrossGroupRoute ids must be globally unique in IR-1",
                path=f"{path}.cross_routes[{route_index}].id",
            )
        result[route.id] = route
    return result


@dataclass(frozen=True, slots=True)
class LogicalRuntimeBinding:
    task_id: str
    flow_id: str | None
    channel_symbol: str | None
    event_symbol: str | None
    token_symbol: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_id, f"{path}.task_id")
        for field_name in ("flow_id", "channel_symbol", "event_symbol", "token_symbol"):
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty(value, f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class CoreOrder:
    core_id: int
    task_ids: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.core_id, f"{path}.core_id")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise SchemaError("contains duplicate task ids", path=f"{path}.task_ids")
        for index, task_id in enumerate(self.task_ids):
            validate_nonempty(task_id, f"{path}.task_ids[{index}]")


@dataclass(frozen=True, slots=True)
class IntraDieSchedule:
    schema_version: str
    producer_pass: str
    id: str
    dag_id: str
    die_id: int
    placements: tuple[TaskPlacement, ...]
    buffer_bindings: tuple[BufferBinding, ...]
    task_buffer_uses: tuple[TaskBufferUse, ...]
    task_state_uses: tuple[TaskStateUse, ...]
    flow_routes: tuple[FlowRouteBinding, ...]
    runtime_bindings: tuple[LogicalRuntimeBinding, ...]
    core_orders: tuple[CoreOrder, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "IntraDieSchedule":
        return cls(
            schema_version=INTRA_DIE_SCHEDULE_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("intra_die_schedule", semantic_key, schema_version=INTRA_DIE_SCHEDULE_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "dag_id", "die_id", "placements", "buffer_bindings", "task_buffer_uses",
            "task_state_uses", "flow_routes", "runtime_bindings", "core_orders",
        )}

    def validate(self, path: str = "intra_die_schedule") -> None:
        if self.schema_version != INTRA_DIE_SCHEDULE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.dag_id, f"{path}.dag_id")
        validate_uint64(self.die_id, f"{path}.die_id")
        placement_tasks: set[str] = set()
        for index, placement in enumerate(self.placements):
            placement.validate(f"{path}.placements[{index}]")
            if placement.task_id in placement_tasks:
                raise SchemaError("duplicate task placement", path=f"{path}.placements[{index}].task_id")
            placement_tasks.add(placement.task_id)
        binding_index = validate_unique_ids(self.buffer_bindings, f"{path}.buffer_bindings")
        for index, binding in enumerate(self.buffer_bindings):
            binding.validate(f"{path}.buffer_bindings[{index}]")
        uses: set[
            tuple[str, str, BufferAccess, BufferUseRole, int, int | None]
        ] = set()
        for index, use in enumerate(self.task_buffer_uses):
            use.validate(f"{path}.task_buffer_uses[{index}]")
            key = (
                use.task_id,
                use.binding_id,
                use.access,
                use.role,
                use.operand_index,
                use.contribution_rank,
            )
            if key in uses:
                raise SchemaError("duplicate task/buffer access", path=f"{path}.task_buffer_uses[{index}]")
            if use.binding_id not in binding_index:
                raise SchemaError("dangling buffer binding", path=f"{path}.task_buffer_uses[{index}].binding_id")
            uses.add(key)
        canonical_uses = tuple(
            sorted(
                self.task_buffer_uses,
                key=lambda use: (
                    use.task_id,
                    _BUFFER_USE_ROLE_ORDER[use.role],
                    use.operand_index,
                    use.contribution_rank
                    if use.contribution_rank is not None
                    else -1,
                    use.binding_id,
                    use.access.value,
                    use.tensor_slice.offset,
                    use.tensor_slice.shape,
                ),
            )
        )
        if self.task_buffer_uses != canonical_uses:
            raise SchemaError(
                "must use canonical task/role/operand order",
                path=f"{path}.task_buffer_uses",
            )
        state_use_tasks: set[str] = set()
        for index, use in enumerate(self.task_state_uses):
            use.validate(f"{path}.task_state_uses[{index}]")
            if use.task_id in state_use_tasks:
                raise SchemaError(
                    "duplicate task state access",
                    path=f"{path}.task_state_uses[{index}].task_id",
                )
            state_use_tasks.add(use.task_id)
        canonical_state_uses = tuple(
            sorted(
                self.task_state_uses,
                key=lambda use: (
                    use.task_id,
                    use.hbm_binding_ref,
                    use.access.value,
                ),
            )
        )
        if self.task_state_uses != canonical_state_uses:
            raise SchemaError(
                "must use canonical task/HBM-binding/access order",
                path=f"{path}.task_state_uses",
            )
        route_flows: set[str] = set()
        for index, route in enumerate(self.flow_routes):
            route.validate(f"{path}.flow_routes[{index}]")
            if route.flow_id in route_flows:
                raise SchemaError("duplicate flow route", path=f"{path}.flow_routes[{index}].flow_id")
            route_flows.add(route.flow_id)
        runtime_tasks: set[str] = set()
        for index, binding in enumerate(self.runtime_bindings):
            binding.validate(f"{path}.runtime_bindings[{index}]")
            if binding.task_id in runtime_tasks:
                raise SchemaError("duplicate runtime binding", path=f"{path}.runtime_bindings[{index}].task_id")
            runtime_tasks.add(binding.task_id)
        cores: set[int] = set()
        ordered_tasks: set[str] = set()
        for index, order in enumerate(self.core_orders):
            order.validate(f"{path}.core_orders[{index}]")
            if order.core_id in cores:
                raise SchemaError("duplicate core order", path=f"{path}.core_orders[{index}].core_id")
            overlap = ordered_tasks.intersection(order.task_ids)
            if overlap:
                raise SchemaError("task appears in multiple core orders", path=f"{path}.core_orders[{index}].task_ids")
            cores.add(order.core_id)
            ordered_tasks.update(order.task_ids)
        expected_id = stable_artifact_id("intra_die_schedule", self._semantic_key(), schema_version=INTRA_DIE_SCHEDULE_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        dag: IntraDieDAG,
        ir1: IR1,
        path: str = "intra_die_schedule",
    ) -> None:
        self.validate(path)
        dag.validate("intra_die_dag")
        ir1.validate("ir1")
        if self.dag_id != dag.id or self.die_id != dag.die_id:
            raise SchemaError("schedule does not identify the supplied DAG/die", path=path)
        if dag.source_ir1_id != ir1.id:
            raise SchemaError("DAG references a different IR-1", path=f"{path}.dag_id")
        dies = {die.id: die for die in ir1.fabric.dies}
        die = dies.get(self.die_id)
        if die is None:
            raise SchemaError("schedule references an unknown die", path=f"{path}.die_id")
        cores = {core.runtime_core_id: core for core in die.cores}
        ports = {port.id: port for port in die.ports}
        route_catalog = _global_route_index(ir1, "ir1")
        task_index = {task.id: task for task in dag.tasks}
        value_index = {value.id: value for value in dag.values}
        staging_value_index = {
            value.id: value for value in dag.state_staging_values
        }
        flow_index = {flow.id: flow for flow in dag.flows}
        executable_tasks = {
            task.id for task in dag.tasks if task.kind is not SemanticTaskKind.TRANSIT
        }
        transit_tasks = set(task_index).difference(executable_tasks)
        placements = {placement.task_id: placement.core_id for placement in self.placements}
        if set(placements) != executable_tasks:
            raise SchemaError(
                "placements must cover every executable task and exclude TRANSIT",
                path=f"{path}.placements",
            )
        if not set(placements.values()).issubset(cores):
            raise SchemaError(
                "placement core_id must name a runtime core on this die",
                path=f"{path}.placements",
            )
        order_by_core = {order.core_id: order.task_ids for order in self.core_orders}
        ordered = {task_id for order in self.core_orders for task_id in order.task_ids}
        if ordered != executable_tasks:
            raise SchemaError(
                "core orders must cover every executable task and exclude TRANSIT",
                path=f"{path}.core_orders",
            )
        if not set(order_by_core).issubset(cores):
            raise SchemaError("core order names a core outside this die", path=f"{path}.core_orders")
        positions: dict[str, int] = {}
        for core_id, task_ids in order_by_core.items():
            for position, task_id in enumerate(task_ids):
                if placements.get(task_id) != core_id:
                    raise SchemaError("core order disagrees with placement", path=f"{path}.core_orders")
                positions[task_id] = position
        for task in dag.tasks:
            for dependency in task.deps:
                if (
                    dependency in executable_tasks
                    and task.id in executable_tasks
                ):
                    if placements[dependency] != placements[task.id]:
                        raise SchemaError(
                            "MVP requires both ends of every executable dependency on the same core",
                            path=f"{path}.placements",
                        )
                    if positions[dependency] >= positions[task.id]:
                        raise SchemaError("core order violates a task dependency", path=f"{path}.core_orders")
        dma_tasks = {
            task.id: task
            for task in dag.tasks
            if task.kind
            in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT)
        }
        state_uses_by_task = {
            use.task_id: use for use in self.task_state_uses
        }
        if set(state_uses_by_task) != set(dma_tasks):
            raise SchemaError(
                "task_state_uses must exactly cover every DMA task and no other task",
                path=f"{path}.task_state_uses",
            )
        manifest = ir1.persistent_state_manifest
        manifest_binding_ids = (
            {binding.id for binding in manifest.bindings}
            if manifest is not None
            else set()
        )
        if manifest_binding_ids.intersection(
            binding.id for binding in self.buffer_bindings
        ):
            raise SchemaError(
                "SRAM and HBM binding id namespaces must be disjoint",
                path=path,
            )
        if dma_tasks:
            if (
                manifest is None
                or dag.source_state_manifest_id != manifest.id
            ):
                raise SchemaError(
                    "state DMA requires the exact IR-1 persistent-state manifest",
                    path=f"{path}.task_state_uses",
                )
            hbm_by_id = {binding.id: binding for binding in manifest.bindings}
            hbm_by_state = {
                binding.state_ref: binding for binding in manifest.bindings
            }
            declaration_by_id = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }
            access_by_id = {access.id: access for access in ir1.state_accesses}
            for task in dma_tasks.values():
                use = state_uses_by_task[task.id]
                origin = task.origin_ref
                contract = task.dma
                assert isinstance(origin, StateIoOrigin)
                assert contract is not None
                access = access_by_id.get(origin.state_access_ref)
                if access is None or (
                    access.node_ref,
                    access.rank,
                    access.state_ref,
                ) != (
                    origin.node_ref,
                    origin.rank,
                    contract.state_ref,
                ):
                    raise SchemaError(
                        "DMA state access disagrees with IR-1 provenance",
                        path=f"{path}.task_state_uses",
                    )
                expected_access = (
                    StateUseAccess.READ
                    if task.kind is SemanticTaskKind.DMA_IN
                    else StateUseAccess.WRITE
                )
                if use.access is not expected_access:
                    raise SchemaError(
                        "DMA_IN must READ HBM and DMA_OUT must WRITE HBM",
                        path=f"{path}.task_state_uses",
                    )
                hbm_binding = hbm_by_id.get(use.hbm_binding_ref)
                expected_binding = hbm_by_state.get(contract.state_ref)
                if (
                    hbm_binding is None
                    or expected_binding is None
                    or hbm_binding.id != expected_binding.id
                ):
                    raise SchemaError(
                        "hbm_binding_ref must name the manifest binding for the DMA state",
                        path=f"{path}.task_state_uses",
                    )
                if (
                    hbm_binding.state_ref,
                    hbm_binding.die_id,
                ) != (
                    contract.state_ref,
                    self.die_id,
                ):
                    raise SchemaError(
                        "HBM binding state/home die disagrees with the DMA payload",
                        path=f"{path}.task_state_uses",
                    )
                if (
                    contract.state_offset_bytes > hbm_binding.size_bytes
                    or task.bytes
                    > hbm_binding.size_bytes - contract.state_offset_bytes
                ):
                    raise SchemaError(
                        "state DMA byte range exceeds its HBM binding",
                        path=f"{path}.task_state_uses",
                    )
                staging = staging_value_index.get(contract.local_value_ref)
                if (
                    staging is None
                    or staging.state_ref != contract.state_ref
                    or task.tensor_slice is None
                    or task.tensor_slice.value_id != staging.id
                ):
                    raise SchemaError(
                        "DMA local endpoint must be its exact state staging value",
                        path=f"{path}.task_state_uses",
                    )
                declaration = declaration_by_id[contract.state_ref]
                for target_ref in contract.access_task_refs:
                    target = task_index[target_ref]
                    if placements[target_ref] != placements[task.id]:
                        raise SchemaError(
                            "DMA and every state-access task must share one core",
                            path=f"{path}.placements",
                        )
                    if task.kind is SemanticTaskKind.DMA_IN:
                        dependency_is_exact = task.id in target.deps
                        order_is_exact = positions[task.id] < positions[target_ref]
                    else:
                        dependency_is_exact = target_ref in task.deps
                        order_is_exact = positions[target_ref] < positions[task.id]
                    if not dependency_is_exact or not order_is_exact:
                        raise SchemaError(
                            "DMA/access dependencies and core order must agree",
                            path=f"{path}.core_orders",
                        )
                    if declaration.identity.tensor_ref is None:
                        compute_values = (
                            tuple(
                                operand.value_id
                                for operand in target.compute.inputs
                                + target.compute.outputs
                            )
                            if target.compute is not None
                            else ()
                        )
                        if staging.id in (
                            target.read_values
                            + target.write_values
                            + compute_values
                        ):
                            raise SchemaError(
                                "opaque state staging must not change target compute arity",
                                path=f"{path}.task_state_uses",
                            )

        if {route.flow_id for route in self.flow_routes} != set(flow_index):
            raise SchemaError("flow routes must cover every DAG flow", path=f"{path}.flow_routes")
        for index, use in enumerate(self.task_buffer_uses):
            if use.task_id in transit_tasks:
                raise SchemaError(
                    "TRANSIT cannot carry a buffer use",
                    path=f"{path}.task_buffer_uses[{index}].task_id",
                )
        binding_index = {binding.id: binding for binding in self.buffer_bindings}
        profile_index = {
            profile.id: profile for profile in ir1.fabric.sram_profiles
        }
        resolved_bindings: dict[
            str,
            tuple[BufferBinding, IntraDieValue | StateStagingValue, int, int],
        ] = {}
        resolved_regions: dict[str, object] = {}
        materializations: set[tuple[int, str]] = set()
        for index, binding in enumerate(self.buffer_bindings):
            binding_path = f"{path}.buffer_bindings[{index}]"
            value = value_index.get(binding.value_id)
            if value is None:
                value = staging_value_index.get(binding.value_id)
            if value is None:
                raise SchemaError("dangling DAG value", path=f"{path}.buffer_bindings[{index}].value_id")
            if binding.core_id not in cores:
                raise SchemaError("buffer core_id lies outside this die", path=f"{path}.buffer_bindings[{index}].core_id")
            if binding.ownership is not BufferOwnership.ALIASED:
                materialization = (binding.core_id, binding.value_id)
                if materialization in materializations:
                    raise SchemaError(
                        "one core/value may have only one root backing",
                        path=f"{binding_path}.value_id",
                    )
                materializations.add(materialization)
            core = cores[binding.core_id]
            profile = profile_index[core.sram_profile_ref]
            regions = {region.id: region for region in profile.regions}
            region = regions.get(binding.region_ref)
            if region is None:
                raise SchemaError("unknown named SRAM region", path=f"{binding_path}.region_ref")
            tensor_slice = binding.tensor_slice
            if len(tensor_slice.shape) != len(value.shape) or any(
                tensor_slice.offset[axis] + tensor_slice.shape[axis]
                > value.shape[axis]
                for axis in range(len(tensor_slice.shape))
            ):
                raise SchemaError("tensor slice lies outside its value", path=f"{binding_path}.tensor_slice")
            if binding.dtype is not value.dtype:
                raise SchemaError("dtype disagrees with DAG value", path=f"{binding_path}.dtype")
            if binding.layout != value.logical_layout:
                raise SchemaError(
                    "layout must equal the DAG value logical_layout in schedule-v1",
                    path=f"{binding_path}.layout",
                )
            element_bytes = {
                DType.FP16: 2,
                DType.FP32: 4,
                DType.INT32: 4,
            }.get(binding.dtype)
            if element_bytes is None:
                raise SchemaError(
                    "unsupported binding dtype", path=f"{binding_path}.dtype"
                )
            element_count = 1
            for dimension in tensor_slice.shape:
                if element_count > UINT64_MAX // dimension:
                    raise SchemaError("tensor slice element count overflows uint64", path=f"{binding_path}.tensor_slice.shape")
                element_count *= dimension
            if element_count > UINT64_MAX // element_bytes:
                raise SchemaError("tensor slice byte count overflows uint64", path=f"{binding_path}.tensor_slice.shape")
            payload_bytes = element_count * element_bytes
            if binding.size_bytes != payload_bytes:
                raise SchemaError(
                    "size_bytes must equal the tight root tensor payload",
                    path=f"{binding_path}.size_bytes",
                )
            if binding.region_offset_bytes + binding.size_bytes > region.size_bytes:
                raise SchemaError("binding span exceeds named SRAM region", path=binding_path)
            absolute_start = region.base_bytes + binding.region_offset_bytes
            absolute_end = absolute_start + binding.size_bytes
            if (
                absolute_start % binding.alignment_bytes != 0
            ):
                raise SchemaError(
                    "binding absolute address does not satisfy alignment_bytes",
                    path=f"{binding_path}.alignment_bytes",
                )
            first_stripe = absolute_start // profile.bank_interleave_bytes
            last_stripe = (absolute_end - 1) // profile.bank_interleave_bytes
            stripe_count = last_stripe - first_stripe + 1
            expected_bank_count = min(stripe_count, profile.bank_count)
            if len(binding.banks) != expected_bank_count or tuple(sorted(binding.banks)) != binding.banks:
                raise SchemaError(
                    "banks do not exactly match the absolute interleave span",
                    path=f"{binding_path}.banks",
                )
            if stripe_count >= profile.bank_count:
                bank_match = all(bank == bank_index for bank_index, bank in enumerate(binding.banks))
            else:
                first_bank = first_stripe % profile.bank_count
                bank_match = all(
                    bank < profile.bank_count
                    and (bank - first_bank) % profile.bank_count < stripe_count
                    for bank in binding.banks
                )
            if not bank_match:
                raise SchemaError(
                    "banks do not exactly match the absolute interleave span",
                    path=f"{binding_path}.banks",
                )
            order = order_by_core.get(binding.core_id)
            if order is None or binding.lifetime_end_exclusive > len(order):
                raise SchemaError("buffer lifetime lies outside its core order", path=binding_path)
            resolved_bindings[binding.id] = (
                binding,
                value,
                absolute_start,
                absolute_end,
            )
            resolved_regions[binding.id] = region
        for index, use in enumerate(self.task_buffer_uses):
            task = task_index.get(use.task_id)
            if task is None:
                raise SchemaError("dangling DAG task", path=f"{path}.task_buffer_uses[{index}].task_id")
            if task.id in transit_tasks:
                raise SchemaError("TRANSIT cannot carry a buffer use", path=f"{path}.task_buffer_uses[{index}].task_id")
            binding = binding_index[use.binding_id]
            if placements[task.id] != binding.core_id:
                raise SchemaError("task and buffer are bound to different cores", path=f"{path}.task_buffer_uses[{index}]")
            if not (
                binding.lifetime_start
                <= positions[task.id]
                < binding.lifetime_end_exclusive
            ):
                raise SchemaError("buffer use lies outside its lifetime", path=f"{path}.task_buffer_uses[{index}]")
            if binding.value_id not in task.read_values + task.write_values:
                raise SchemaError("task does not access the bound value", path=f"{path}.task_buffer_uses[{index}]")
            if use.tensor_slice.value_id != binding.value_id:
                raise SchemaError(
                    "buffer-use view references a different value",
                    path=f"{path}.task_buffer_uses[{index}].tensor_slice.value_id",
                )
            dense_row_major_view_byte_addend(
                binding.tensor_slice,
                use.tensor_slice,
                binding.dtype,
                path=f"{path}.task_buffer_uses[{index}].tensor_slice",
            )
        uses_by_task: dict[str, list[TaskBufferUse]] = {
            task.id: [] for task in dag.tasks
        }
        for use in self.task_buffer_uses:
            uses_by_task[use.task_id].append(use)

        def expected_uses(
            task: SemanticTask,
        ) -> tuple[
            tuple[BufferUseRole, int, int | None, str, BufferAccess], ...
        ]:
            if task.kind is SemanticTaskKind.COMP:
                assert task.compute is not None
                input_roles, output_roles = canonical_compute_operand_roles(
                    task.compute.op_kind,
                    task.compute.workload,
                    tiled=task.compute.tile is not None,
                    path=f"{path}.tasks.{task.id}.compute",
                )
                if (
                    task.op_kind is not task.compute.op_kind
                    or tuple(operand.role for operand in task.compute.inputs)
                    != input_roles
                    or tuple(operand.role for operand in task.compute.outputs)
                    != output_roles
                ):
                    raise SchemaError(
                        "COMP task and ComputeContract op/workload/roles are not closed",
                        path=f"{path}.task_buffer_uses",
                    )
                return tuple(
                    (
                        BufferUseRole.COMP_INPUT,
                        index,
                        None,
                        operand.value_id,
                        BufferAccess.READ,
                    )
                    for index, operand in enumerate(task.compute.inputs)
                ) + tuple(
                    (
                        BufferUseRole.COMP_OUTPUT,
                        index,
                        None,
                        operand.value_id,
                        BufferAccess.WRITE,
                    )
                    for index, operand in enumerate(task.compute.outputs)
                )
            if task.kind is SemanticTaskKind.SEND:
                if len(task.read_values) != 1 or task.write_values:
                    raise SchemaError(
                        "SEND requires exactly one source operand and no outputs",
                        path=f"{path}.task_buffer_uses",
                    )
                return tuple(
                    (BufferUseRole.SEND_SOURCE, index, None, value_id, BufferAccess.READ)
                    for index, value_id in enumerate(task.read_values)
                )
            if task.kind is SemanticTaskKind.RECV:
                if task.read_values or len(task.write_values) != 1:
                    raise SchemaError(
                        "RECV requires no inputs and exactly one destination operand",
                        path=f"{path}.task_buffer_uses",
                    )
                return tuple(
                    (
                        BufferUseRole.RECV_DESTINATION,
                        index,
                        None,
                        value_id,
                        BufferAccess.WRITE,
                    )
                    for index, value_id in enumerate(task.write_values)
                )
            if task.kind is SemanticTaskKind.REDUCE:
                assert task.reduction is not None
                if len(task.read_values) != len(task.reduction.input_ranks):
                    raise SchemaError(
                        "REDUCE read operands must exactly match input_ranks",
                        path=f"{path}.task_buffer_uses",
                    )
                return tuple(
                    (
                        BufferUseRole.REDUCE_INPUT,
                        index,
                        rank,
                        task.read_values[index],
                        BufferAccess.READ,
                    )
                    for index, rank in enumerate(task.reduction.input_ranks)
                ) + tuple(
                    (
                        BufferUseRole.REDUCE_OUTPUT,
                        index,
                        None,
                        value_id,
                        BufferAccess.WRITE,
                    )
                    for index, value_id in enumerate(task.write_values)
                )
            if task.kind is SemanticTaskKind.LOCAL_COPY:
                if len(task.read_values) != 1 or len(task.write_values) != 1:
                    raise SchemaError(
                        "LOCAL_COPY requires exactly one source and one destination operand",
                        path=f"{path}.task_buffer_uses",
                    )
                return tuple(
                    (
                        BufferUseRole.LOCAL_COPY_SOURCE,
                        index,
                        None,
                        value_id,
                        BufferAccess.READ,
                    )
                    for index, value_id in enumerate(task.read_values)
                ) + tuple(
                    (
                        BufferUseRole.LOCAL_COPY_DESTINATION,
                        index,
                        None,
                        value_id,
                        BufferAccess.WRITE,
                    )
                    for index, value_id in enumerate(task.write_values)
                )
            if task.kind is SemanticTaskKind.DMA_IN:
                assert task.dma is not None
                if task.read_values or task.write_values != (
                    task.dma.local_value_ref,
                ):
                    raise SchemaError(
                        "DMA_IN requires exactly one local SRAM destination",
                        path=f"{path}.task_buffer_uses",
                    )
                return (
                    (
                        BufferUseRole.DMA_DESTINATION,
                        0,
                        None,
                        task.dma.local_value_ref,
                        BufferAccess.WRITE,
                    ),
                )
            if task.kind is SemanticTaskKind.DMA_OUT:
                assert task.dma is not None
                if task.read_values != (
                    task.dma.local_value_ref,
                ) or task.write_values:
                    raise SchemaError(
                        "DMA_OUT requires exactly one local SRAM source",
                        path=f"{path}.task_buffer_uses",
                    )
                return (
                    (
                        BufferUseRole.DMA_SOURCE,
                        0,
                        None,
                        task.dma.local_value_ref,
                        BufferAccess.READ,
                    ),
                )
            return ()

        role_initiator = {
            BufferUseRole.COMP_INPUT: MemoryInitiator.COMPUTE,
            BufferUseRole.COMP_OUTPUT: MemoryInitiator.COMPUTE,
            BufferUseRole.SEND_SOURCE: MemoryInitiator.DTE,
            BufferUseRole.RECV_DESTINATION: MemoryInitiator.NOC_RX,
            BufferUseRole.REDUCE_INPUT: MemoryInitiator.COMPUTE,
            BufferUseRole.REDUCE_OUTPUT: MemoryInitiator.COMPUTE,
            BufferUseRole.LOCAL_COPY_SOURCE: MemoryInitiator.LSU,
            BufferUseRole.LOCAL_COPY_DESTINATION: MemoryInitiator.LSU,
            BufferUseRole.DMA_SOURCE: MemoryInitiator.LSU,
            BufferUseRole.DMA_DESTINATION: MemoryInitiator.LSU,
        }
        slice_bound_kinds = {
            SemanticTaskKind.SEND,
            SemanticTaskKind.RECV,
            SemanticTaskKind.REDUCE,
            SemanticTaskKind.LOCAL_COPY,
            SemanticTaskKind.DMA_IN,
            SemanticTaskKind.DMA_OUT,
        }
        for task in dag.tasks:
            expected = expected_uses(task)
            actual = uses_by_task[task.id]
            actual_by_operand = {
                (use.role, use.operand_index): use for use in actual
            }
            expected_keys = {(role, index) for role, index, _rank, _value, _access in expected}
            if len(actual_by_operand) != len(actual) or set(actual_by_operand) != expected_keys:
                raise SchemaError(
                    f"task {task.id!r} buffer uses must exactly cover required roles/operands",
                    path=f"{path}.task_buffer_uses",
                )
            if task.kind in slice_bound_kinds:
                if task.tensor_slice is None or task.dtype is None:
                    raise SchemaError(
                        "payload task requires a tensor slice and dtype",
                        path=f"{path}.task_buffer_uses",
                    )
                element_bytes = {
                    DType.FP16: 2,
                    DType.FP32: 4,
                    DType.INT32: 4,
                }.get(task.dtype)
                if element_bytes is None:
                    raise SchemaError(
                        "unsupported payload dtype",
                        path=f"{path}.task_buffer_uses",
                    )
                tight_bytes = math.prod(task.tensor_slice.shape) * element_bytes
                if tight_bytes > UINT64_MAX or task.bytes != tight_bytes:
                    raise SchemaError(
                        "task.bytes must equal the tight tensor-slice payload",
                        path=f"{path}.task_buffer_uses",
                    )
            for role, operand_index, rank, value_id, access in expected:
                use = actual_by_operand[(role, operand_index)]
                binding = binding_index[use.binding_id]
                if (
                    use.contribution_rank != rank
                    or use.access is not access
                    or binding.value_id != value_id
                ):
                    raise SchemaError(
                        "buffer use rank/access/value disagrees with required operand",
                        path=f"{path}.task_buffer_uses",
                    )
                region = resolved_regions[binding.id]
                if role_initiator[role] not in region.access:
                    raise SchemaError(
                        f"named SRAM region does not permit {role_initiator[role].value}",
                        path=f"{path}.task_buffer_uses",
                    )
                if task.kind in slice_bound_kinds and (
                    use.tensor_slice.offset != task.tensor_slice.offset
                    or use.tensor_slice.shape != task.tensor_slice.shape
                ):
                    raise SchemaError(
                        "buffer-use view geometry (offset, shape) must equal task.tensor_slice; value_id remains operand-specific",
                        path=f"{path}.task_buffer_uses",
                    )
        def transfer_buffer_use(
            task: SemanticTask,
            role: BufferUseRole,
        ) -> TaskBufferUse:
            matches = tuple(
                use for use in uses_by_task[task.id] if use.role is role
            )
            if len(matches) != 1:
                raise SchemaError(
                    "state-transfer endpoint requires one exact local staging use",
                    path=f"{path}.task_buffer_uses",
                )
            return matches[0]

        local_transfer_tasks: dict[
            tuple[str, int | None], list[SemanticTask]
        ] = {}
        for task in dag.tasks:
            if isinstance(task.origin_ref, StateTransferOrigin):
                local_transfer_tasks.setdefault(
                    (
                        task.origin_ref.state_transfer_ref,
                        task.origin_ref.segment_index,
                    ),
                    [],
                ).append(task)
        if {
            transfer_ref for transfer_ref, _segment_index in local_transfer_tasks
        } != set(dag.state_transfer_ids):
            raise SchemaError(
                "scheduled state-transfer tasks must exactly cover local transfer ids",
                path=f"{path}.placements",
            )
        runtime_by_task = {
            binding.task_id: binding for binding in self.runtime_bindings
        }
        ordered_transfer_units: list[tuple[str, int | None]] = []
        for transfer_ref in dag.state_transfer_ids:
            segment_indices = tuple(
                sorted(
                    (
                        segment_index
                        for candidate_ref, segment_index in local_transfer_tasks
                        if candidate_ref == transfer_ref
                    ),
                    key=lambda item: -1 if item is None else item,
                )
            )
            if segment_indices != (None,) and segment_indices != tuple(
                range(len(segment_indices))
            ):
                raise SchemaError(
                    "segmented state-transfer units must use contiguous canonical indices",
                    path=f"{path}.placements",
                )
            ordered_transfer_units.extend(
                (transfer_ref, segment_index)
                for segment_index in segment_indices
            )
        for transfer_ref, segment_index in ordered_transfer_units:
            transfer_tasks = tuple(
                local_transfer_tasks[(transfer_ref, segment_index)]
            )
            kinds = tuple(task.kind for task in transfer_tasks)
            if kinds == (SemanticTaskKind.TRANSIT,):
                continue
            if kinds == (SemanticTaskKind.SEND,):
                send = transfer_tasks[0]
                staging_id = send.read_values[0]
                endpoint_dmas = tuple(
                    task
                    for task in dma_tasks.values()
                    if (
                        task.dma is not None
                        and task.dma.local_value_ref == staging_id
                    )
                )
                if len(endpoint_dmas) != 1:
                    raise SchemaError(
                        "state-transfer source requires one exact state DMA endpoint",
                        path=f"{path}.task_state_uses",
                    )
                dma = endpoint_dmas[0]
                assert dma.dma is not None
                targets = tuple(
                    task_index[target_ref]
                    for target_ref in dma.dma.access_task_refs
                )
                legacy_read_chain = (
                    dma.kind is SemanticTaskKind.DMA_IN
                    and not dma.deps
                    and all(
                        target.kind is SemanticTaskKind.COMP
                        and dma.id in target.deps
                        for target in targets
                    )
                )
                sliced_write_chain = (
                    dma.kind is SemanticTaskKind.DMA_OUT
                    and dma.deps == dma.dma.access_task_refs
                    and all(
                        target.kind is SemanticTaskKind.COMP
                        and dma.id not in target.deps
                        for target in targets
                    )
                )
                if (
                    not targets
                    or not (legacy_read_chain or sliced_write_chain)
                    or send.deps != dma.dma.access_task_refs
                ):
                    raise SchemaError(
                        "state-transfer source must be DMA_IN -> COMP -> SEND or COMP -> SEND/DMA_OUT",
                        path=f"{path}.core_orders",
                    )
                dma_role = (
                    BufferUseRole.DMA_DESTINATION
                    if legacy_read_chain
                    else BufferUseRole.DMA_SOURCE
                )
                if (
                    transfer_buffer_use(
                        dma, dma_role
                    ).binding_id
                    != transfer_buffer_use(
                        send, BufferUseRole.SEND_SOURCE
                    ).binding_id
                ):
                    raise SchemaError(
                        "state-transfer source state DMA and SEND must reuse one staging root",
                        path=f"{path}.task_buffer_uses",
                    )
                component = (
                    (dma, *targets, send)
                    if legacy_read_chain
                    else (*targets, send, dma)
                )
                required_initiators = {
                    MemoryInitiator.LSU,
                    MemoryInitiator.COMPUTE,
                    MemoryInitiator.DTE,
                }
            elif (
                len(transfer_tasks) == 2
                and {task.kind for task in transfer_tasks}
                == {SemanticTaskKind.RECV, SemanticTaskKind.WAIT}
            ):
                recv = next(
                    task
                    for task in transfer_tasks
                    if task.kind is SemanticTaskKind.RECV
                )
                wait = next(
                    task
                    for task in transfer_tasks
                    if task.kind is SemanticTaskKind.WAIT
                )
                staging_id = recv.write_values[0]
                endpoint_dmas = tuple(
                    task
                    for task in dma_tasks.values()
                    if (
                        task.kind is SemanticTaskKind.DMA_OUT
                        and task.dma is not None
                        and task.dma.local_value_ref == staging_id
                    )
                )
                if len(endpoint_dmas) != 1:
                    raise SchemaError(
                        "state-transfer destination requires one exact DMA_OUT endpoint",
                        path=f"{path}.task_state_uses",
                    )
                dma = endpoint_dmas[0]
                assert dma.dma is not None
                targets = tuple(
                    task_index[target_ref]
                    for target_ref in dma.dma.access_task_refs
                )
                if (
                    not targets
                    or wait.deps != (recv.id,)
                    or any(
                        target.kind is not SemanticTaskKind.COMP
                        or wait.id not in target.deps
                        for target in targets
                    )
                    or dma.deps != dma.dma.access_task_refs
                ):
                    raise SchemaError(
                        "state-transfer destination must be RECV -> WAIT -> COMP -> DMA_OUT",
                        path=f"{path}.core_orders",
                    )
                if (
                    transfer_buffer_use(
                        recv, BufferUseRole.RECV_DESTINATION
                    ).binding_id
                    != transfer_buffer_use(
                        dma, BufferUseRole.DMA_SOURCE
                    ).binding_id
                ):
                    raise SchemaError(
                        "state-transfer RECV and DMA_OUT must reuse one staging root",
                        path=f"{path}.task_buffer_uses",
                    )
                recv_runtime = runtime_by_task.get(recv.id)
                wait_runtime = runtime_by_task.get(wait.id)
                if (
                    recv_runtime is None
                    or wait_runtime is None
                    or recv_runtime.token_symbol != wait_runtime.token_symbol
                    or recv_runtime.event_symbol != wait_runtime.event_symbol
                ):
                    raise SchemaError(
                        "state-transfer WAIT must reuse its RECV event and token",
                        path=f"{path}.runtime_bindings",
                    )
                component = (recv, wait, *targets, dma)
                required_initiators = {
                    MemoryInitiator.NOC_RX,
                    MemoryInitiator.COMPUTE,
                    MemoryInitiator.LSU,
                }
            else:
                raise SchemaError(
                    "state-transfer unit must contain exactly SEND, RECV+WAIT, or TRANSIT",
                    path=f"{path}.placements",
                )
            component_core_ids = {
                placements[task.id] for task in component
            }
            if len(component_core_ids) != 1:
                raise SchemaError(
                    "state-transfer endpoint component must share one core",
                    path=f"{path}.placements",
                )
            component_core = cores[next(iter(component_core_ids))]
            component_profile = profile_index[
                component_core.sram_profile_ref
            ]
            if not any(
                required_initiators.issubset(region.access)
                for region in component_profile.regions
            ):
                raise SchemaError(
                    "state-transfer endpoint core lacks one SRAM region for all required initiators",
                    path=f"{path}.placements",
                )
        uses_by_binding: dict[str, list[TaskBufferUse]] = {
            binding.id: [] for binding in self.buffer_bindings
        }
        for use in self.task_buffer_uses:
            uses_by_binding[use.binding_id].append(use)
        for binding_index_value, binding in enumerate(self.buffer_bindings):
            binding_path = f"{path}.buffer_bindings[{binding_index_value}]"
            binding_uses = uses_by_binding[binding.id]
            if binding.ownership is BufferOwnership.ALIASED:
                continue
            if not binding_uses:
                raise SchemaError(
                    "root backing must have at least one task use",
                    path=binding_path,
                )
            rank = len(binding_uses[0].tensor_slice.shape)
            minimum = tuple(
                min(use.tensor_slice.offset[axis] for use in binding_uses)
                for axis in range(rank)
            )
            maximum = tuple(
                max(
                    use.tensor_slice.offset[axis]
                    + use.tensor_slice.shape[axis]
                    for use in binding_uses
                )
                for axis in range(rank)
            )
            expected_root = TensorSlice(
                binding.value_id,
                minimum,
                tuple(
                    maximum[axis] - minimum[axis] for axis in range(rank)
                ),
            )
            if binding.tensor_slice != expected_root:
                raise SchemaError(
                    "root backing must be the exact minimum rectangular cover of all operand views",
                    path=f"{binding_path}.tensor_slice",
                )
            use_positions = tuple(
                positions[use.task_id] for use in binding_uses
            )
            if (
                binding.lifetime_start != min(use_positions)
                or binding.lifetime_end_exclusive != max(use_positions) + 1
            ):
                raise SchemaError(
                    "root lifetime must exactly span all operand uses",
                    path=binding_path,
                )
            if binding.ownership is not BufferOwnership.ALIASED:
                expected_ownership = (
                    BufferOwnership.OWNED
                    if any(use.access is BufferAccess.WRITE for use in binding_uses)
                    else BufferOwnership.BORROWED
                )
                if binding.ownership is not expected_ownership:
                    raise SchemaError(
                        "root ownership must derive from whether this storage has a writer",
                        path=f"{binding_path}.ownership",
                    )
        for task in dag.tasks:
            if task.kind is not SemanticTaskKind.REDUCE:
                continue
            assert task.reduction is not None
            if len(task.reduction.input_ranks) > (1 << 16) - 1:
                raise SchemaError(
                    "LOCAL_REDUCE input_count must fit uint16",
                    path=f"{path}.task_buffer_uses",
                )
            inputs = tuple(
                binding_index[use.binding_id]
                for use in uses_by_task[task.id]
                if use.role is BufferUseRole.REDUCE_INPUT
            )
            outputs = tuple(
                binding_index[use.binding_id]
                for use in uses_by_task[task.id]
                if use.role is BufferUseRole.REDUCE_OUTPUT
            )
            if len(outputs) != 1:
                raise SchemaError(
                    "LOCAL_REDUCE requires exactly one output",
                    path=f"{path}.task_buffer_uses",
                )
            output = outputs[0]
            if any(
                binding.dtype is not DType.FP16 for binding in inputs + outputs
            ):
                raise SchemaError(
                    "LOCAL_REDUCE bindings must be FP16",
                    path=f"{path}.task_buffer_uses",
                )
            core_ids = {binding.core_id for binding in inputs + outputs}
            region_refs = {binding.region_ref for binding in inputs + outputs}
            if len(core_ids) != 1 or len(region_refs) != 1:
                raise SchemaError(
                    "LOCAL_REDUCE inputs/output must share one core and named region",
                    path=f"{path}.task_buffer_uses",
                )
            chunk_bytes = task.bytes
            source_base = inputs[0].region_offset_bytes
            if any(
                binding.region_offset_bytes != source_base + index * chunk_bytes
                or binding.size_bytes != chunk_bytes
                for index, binding in enumerate(inputs)
            ) or output.size_bytes != chunk_bytes:
                raise SchemaError(
                    "LOCAL_REDUCE inputs must be rank-ordered tight-stride chunks",
                    path=f"{path}.task_buffer_uses",
                )
            region = resolved_regions[inputs[0].id]
            source_end = source_base + len(inputs) * chunk_bytes
            if source_end > region.size_bytes:
                raise SchemaError(
                    "LOCAL_REDUCE source span exceeds named region",
                    path=f"{path}.task_buffer_uses",
                )
            output_start = output.region_offset_bytes
            output_end = output_start + output.size_bytes
            if output_start < source_end and source_base < output_end:
                raise SchemaError(
                    "LOCAL_REDUCE output must not overlap the source span",
                    path=f"{path}.task_buffer_uses",
                )
            absolute_source = region.base_bytes + source_base
            output_region = resolved_regions[output.id]
            absolute_output = output_region.base_bytes + output_start
            if absolute_source % 2 or absolute_output % 2:
                raise SchemaError(
                    "LOCAL_REDUCE source/output must be 2-byte aligned",
                    path=f"{path}.task_buffer_uses",
                )
        storage_groups: dict[str, list[BufferBinding]] = {}
        for index, binding in enumerate(self.buffer_bindings):
            binding_path = f"{path}.buffer_bindings[{index}]"
            storage_groups.setdefault(binding.storage_id, []).append(binding)
            if binding.ownership is not BufferOwnership.ALIASED:
                continue
            root = binding_index.get(binding.alias_of or "")
            if root is None:
                raise SchemaError("alias_of references an unknown root", path=f"{binding_path}.alias_of")
            if root.ownership is BufferOwnership.ALIASED or root.alias_of is not None:
                raise SchemaError("alias_of must directly reference a canonical root", path=f"{binding_path}.alias_of")
            if (
                binding.core_id,
                binding.region_ref,
                binding.region_offset_bytes,
                binding.size_bytes,
                binding.storage_id,
            ) != (
                root.core_id,
                root.region_ref,
                root.region_offset_bytes,
                root.size_bytes,
                root.storage_id,
            ):
                raise SchemaError(
                    "alias and root must have identical core/region/span/storage",
                    path=binding_path,
                )
            alias_value = value_index.get(binding.value_id)
            root_value = value_index.get(root.value_id)
            if alias_value is None:
                raise SchemaError(
                    "alias binding must reference an ordinary output value",
                    path=f"{binding_path}.value_id",
                )
            if root_value is not None:
                if (
                    alias_value.alias_set is None
                    or alias_value.alias_set != root_value.alias_set
                ):
                    raise SchemaError(
                        "alias values must share one explicit alias_set",
                        path=f"{binding_path}.value_id",
                    )
                continue

            staging = staging_value_index.get(root.value_id)
            if staging is None or manifest is None:
                raise SchemaError(
                    "alias root must resolve to an ordinary or trainable-state value",
                    path=f"{binding_path}.alias_of",
                )
            declaration = next(
                (
                    item
                    for item in manifest.declarations
                    if item.id == staging.state_ref
                ),
                None,
            )
            access = next(
                (
                    item
                    for item in ir1.state_accesses
                    if item.id == staging.state_access_ref
                ),
                None,
            )
            alias_uses = tuple(
                use
                for uses in uses_by_task.values()
                for use in uses
                if use.binding_id == binding.id
            )
            root_uses = tuple(
                use
                for uses in uses_by_task.values()
                for use in uses
                if use.binding_id == root.id
            )
            if len(alias_uses) != 1:
                raise SchemaError(
                    "trainable-state alias must have one optimizer output use",
                    path=f"{binding_path}.value_id",
                )
            alias_use = alias_uses[0]
            optimizer = task_index[alias_use.task_id]
            matching_root_uses = tuple(
                use
                for use in root_uses
                if use.task_id == optimizer.id
                and use.role is BufferUseRole.COMP_INPUT
                and use.operand_index == 0
            )
            if (
                declaration is None
                or access is None
                or declaration.identity.kind
                is not StateKind.TRAINABLE_PARAMETER
                or declaration.lifetime
                is not PersistentStateLifetime.PERSISTENT
                or declaration.access is not PersistentStateAccess.READ_WRITE
                or declaration.identity.tensor_ref is None
                or access.state_ref != declaration.id
                or access.mode is not StateAccessMode.READ_WRITE
                or optimizer.kind is not SemanticTaskKind.COMP
                or optimizer.op_kind is not OpKind.OPTIMIZER_UPDATE
                or not isinstance(optimizer.origin_ref, OrdinaryNodeOrigin)
                or access.node_ref != optimizer.origin_ref.op_id
                or optimizer.compute is None
                or optimizer.compute.effects.alias_set != alias_value.alias_set
                or alias_value.alias_set
                != f"trainable:{declaration.identity.tensor_ref}"
                or tuple(operand.value_id for operand in optimizer.compute.inputs)
                != (root.value_id, optimizer.read_values[1])
                or tuple(operand.value_id for operand in optimizer.compute.outputs)
                != (binding.value_id,)
                or alias_use.role is not BufferUseRole.COMP_OUTPUT
                or alias_use.operand_index != 0
                or len(matching_root_uses) != 1
                or binding.tensor_slice.offset != root.tensor_slice.offset
                or binding.tensor_slice.shape != root.tensor_slice.shape
            ):
                raise SchemaError(
                    "state staging may alias only one exact READ_WRITE trainable optimizer update",
                    path=f"{binding_path}.value_id",
                )
        for storage_id, members in storage_groups.items():
            if len(members) == 1:
                continue
            roots = tuple(
                binding
                for binding in members
                if binding.ownership is not BufferOwnership.ALIASED
            )
            if len(roots) != 1 or any(
                binding.ownership is BufferOwnership.ALIASED
                and binding.alias_of != roots[0].id
                for binding in members
            ):
                raise SchemaError(
                    f"storage_id {storage_id!r} may only be reused by aliases of one canonical root",
                    path=f"{path}.buffer_bindings",
                )
        for left_index, left in enumerate(self.buffer_bindings):
            _left_binding, _left_value, left_start, left_end = resolved_bindings[left.id]
            for right_index in range(left_index + 1, len(self.buffer_bindings)):
                right = self.buffer_bindings[right_index]
                _right_binding, _right_value, right_start, right_end = resolved_bindings[right.id]
                byte_overlap = left_start < right_end and right_start < left_end
                lifetime_overlap = (
                    left.lifetime_start < right.lifetime_end_exclusive
                    and right.lifetime_start < left.lifetime_end_exclusive
                )
                if (
                    left.core_id == right.core_id
                    and byte_overlap
                    and lifetime_overlap
                    and left.storage_id != right.storage_id
                ):
                    raise SchemaError(
                        f"physical byte/lifetime overlap with buffer_bindings[{left_index}]",
                        path=f"{path}.buffer_bindings[{right_index}]",
                    )
        runtime = {binding.task_id: binding for binding in self.runtime_bindings}
        if not set(runtime).issubset(task_index):
            raise SchemaError("runtime binding references a dangling task", path=f"{path}.runtime_bindings")
        if set(runtime).intersection(transit_tasks):
            raise SchemaError("TRANSIT cannot carry a runtime binding", path=f"{path}.runtime_bindings")
        runtime_kinds = (
            SemanticTaskKind.SEND,
            SemanticTaskKind.RECV,
            SemanticTaskKind.WAIT,
            SemanticTaskKind.BARRIER,
        )
        expected_runtime_tasks = {
            task.id for task in dag.tasks if task.kind in runtime_kinds
        }
        if set(runtime) != expected_runtime_tasks:
            raise SchemaError(
                "runtime bindings must exactly cover SEND/RECV/WAIT/BARRIER tasks",
                path=f"{path}.runtime_bindings",
            )
        for task in dag.tasks:
            if task.kind in runtime_kinds:
                binding = runtime.get(task.id)
                assert binding is not None
                if binding.token_symbol is None:
                    raise SchemaError(
                        "runtime binding requires a nonempty token symbol",
                        path=f"{path}.runtime_bindings",
                    )
                if task.sync is None:
                    raise SchemaError(
                        "runtime task requires a sync contract",
                        path=f"{path}.runtime_bindings",
                    )
                if task.kind in (SemanticTaskKind.SEND, SemanticTaskKind.RECV):
                    flow = flow_index.get(task.flow_id or "")
                    if (
                        flow is None
                        or binding.flow_id != task.flow_id
                        or binding.channel_symbol != flow.logical_channel
                        or binding.event_symbol != task.sync.completion_event
                    ):
                        raise SchemaError(
                            "transport runtime binding must preserve flow, channel, and completion event",
                            path=f"{path}.runtime_bindings",
                        )
                elif task.kind is SemanticTaskKind.WAIT:
                    if (
                        binding.flow_id is not None
                        or binding.channel_symbol is not None
                        or binding.event_symbol != task.sync.wait_event
                    ):
                        raise SchemaError(
                            "WAIT runtime binding must reference exactly its wait event",
                            path=f"{path}.runtime_bindings",
                        )
                    waited_recvs = tuple(
                        task_index[dependency]
                        for dependency in task.deps
                        if task_index[dependency].kind is SemanticTaskKind.RECV
                        and task_index[dependency].origin_ref.rank
                        == task.origin_ref.rank
                        and task_index[dependency].sync is not None
                        and task_index[dependency].sync.completion_event
                        == task.sync.wait_event
                    )
                    if len(waited_recvs) != 1:
                        raise SchemaError(
                            "WAIT must identify exactly one same-rank RECV dependency by wait event",
                            path=f"{path}.runtime_bindings",
                        )
                    recv_binding = runtime[waited_recvs[0].id]
                    if binding.token_symbol != recv_binding.token_symbol:
                        raise SchemaError(
                            "WAIT must reuse its RECV runtime token symbol",
                            path=f"{path}.runtime_bindings",
                        )
                else:
                    barrier = task.sync.barrier
                    if (
                        barrier is None
                        or barrier.scope is not BarrierScope.PLAN
                        or binding.flow_id is not None
                        or binding.channel_symbol is not None
                        or binding.event_symbol != barrier.id
                    ):
                        raise SchemaError(
                            "PLAN BARRIER runtime binding must reference exactly its shared barrier id",
                            path=f"{path}.runtime_bindings",
                        )
        scheduled_routes = {binding.flow_id: binding for binding in self.flow_routes}
        for flow_index_value, flow in enumerate(dag.flows):
            route_path = f"{path}.flow_routes[{flow_index_value}]"
            binding = scheduled_routes[flow.id]
            if binding.pair_route_ref != flow.pair_route_ref:
                raise SchemaError("pair_route_ref disagrees with SemanticFlow", path=f"{route_path}.pair_route_ref")
            route = route_catalog.get(flow.pair_route_ref)
            if route is None:
                raise SchemaError("SemanticFlow references an unknown route", path=f"{route_path}.pair_route_ref")
            if (
                route.source_rank,
                route.destination_rank,
                route.die_path,
            ) != (
                flow.source_rank,
                flow.destination_rank,
                flow.die_path,
            ):
                raise SchemaError("SemanticFlow disagrees with PairRoute endpoints/path", path=route_path)
            die_position = route.die_path.index(self.die_id)
            if die_position == 0:
                expected_role = FlowRouteRole.SOURCE
                expected_ingress = None
                first_hop = route.hops[0]
                expected_egress = PortLeg(first_hop.link_ref, first_hop.source_port_ref)
                task_kind = SemanticTaskKind.SEND
            elif die_position == len(route.die_path) - 1:
                expected_role = FlowRouteRole.DESTINATION
                last_hop = route.hops[-1]
                expected_ingress = PortLeg(last_hop.link_ref, last_hop.destination_port_ref)
                expected_egress = None
                task_kind = SemanticTaskKind.RECV
            else:
                expected_role = FlowRouteRole.TRANSIT
                ingress_hop = route.hops[die_position - 1]
                egress_hop = route.hops[die_position]
                expected_ingress = PortLeg(
                    ingress_hop.link_ref, ingress_hop.destination_port_ref
                )
                expected_egress = PortLeg(
                    egress_hop.link_ref, egress_hop.source_port_ref
                )
                task_kind = SemanticTaskKind.TRANSIT
            if (
                binding.role,
                binding.ingress,
                binding.egress,
            ) != (expected_role, expected_ingress, expected_egress):
                raise SchemaError("route role or port legs disagree with PairRoute hop", path=route_path)
            local_tasks = tuple(task_index[task_id] for task_id in flow.task_ids)
            if len(local_tasks) != 1 or local_tasks[0].kind is not task_kind:
                raise SchemaError("route role disagrees with local transport task", path=route_path)
            if expected_role is FlowRouteRole.SOURCE:
                start = cores[placements[local_tasks[0].id]].noc_coord
                end = ports[expected_egress.port_ref].noc_coord
            elif expected_role is FlowRouteRole.DESTINATION:
                start = ports[expected_ingress.port_ref].noc_coord
                end = cores[placements[local_tasks[0].id]].noc_coord
            else:
                start = ports[expected_ingress.port_ref].noc_coord
                end = ports[expected_egress.port_ref].noc_coord
            if binding.local_noc_path != _backend_xy_noc_path(start, end):
                raise SchemaError(
                    "local_noc_path must be the exact backend-v1 X-then-Y path",
                    path=f"{route_path}.local_noc_path",
                )


@dataclass(frozen=True, slots=True)
class IntraDieScheduleSet:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    source_ir1_id: str
    schedules: tuple[IntraDieSchedule, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "IntraDieScheduleSet":
        return cls(
            schema_version=INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "intra_die_schedule_set",
                semantic_key,
                schema_version=INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_projection_id": self.source_projection_id,
            "source_ir1_id": self.source_ir1_id,
            "schedules": self.schedules,
        }

    def validate(self, path: str = "intra_die_schedule_set") -> None:
        if self.schema_version != INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        if not self.schedules:
            raise SchemaError("must contain schedules", path=f"{path}.schedules")
        schedule_ids: set[str] = set()
        dag_ids: set[str] = set()
        die_ids: set[int] = set()
        for index, schedule in enumerate(self.schedules):
            schedule.validate(f"{path}.schedules[{index}]")
            if schedule.id in schedule_ids or schedule.dag_id in dag_ids or schedule.die_id in die_ids:
                raise SchemaError(
                    "schedule id, dag_id, and die_id must each be unique",
                    path=f"{path}.schedules[{index}]",
                )
            schedule_ids.add(schedule.id)
            dag_ids.add(schedule.dag_id)
            die_ids.add(schedule.die_id)
        expected_id = stable_artifact_id(
            "intra_die_schedule_set",
            self._semantic_key(),
            schema_version=INTRA_DIE_SCHEDULE_SET_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
        path: str = "intra_die_schedule_set",
    ) -> None:
        self.validate(path)
        projection.validate("ir2_projection_result")
        ir1.validate("ir1")
        if self.source_projection_id != projection.id:
            raise SchemaError("schedule set references a different projection", path=f"{path}.source_projection_id")
        if self.source_ir1_id != ir1.id or projection.source_ir1_id != ir1.id:
            raise SchemaError("schedule set/projection reference a different IR-1", path=f"{path}.source_ir1_id")
        dag_index = {dag.id: dag for dag in projection.dags}
        if {schedule.dag_id for schedule in self.schedules} != set(dag_index):
            raise SchemaError("must contain exactly one schedule for every projection DAG", path=f"{path}.schedules")
        if tuple(
            (schedule.dag_id, schedule.die_id) for schedule in self.schedules
        ) != tuple((dag.id, dag.die_id) for dag in projection.dags):
            raise SchemaError(
                "schedules must exactly follow projection DAG tuple order",
                path=f"{path}.schedules",
            )
        occurrences: dict[str, list[tuple[int, SemanticFlow, FlowRouteBinding]]] = {}
        barrier_definitions: dict[str, BarrierContract] = {}
        barrier_ranks: dict[str, list[int]] = {}
        for index, schedule in enumerate(self.schedules):
            dag = dag_index[schedule.dag_id]
            schedule.validate_against(dag, ir1, f"{path}.schedules[{index}]")
            bindings = {binding.flow_id: binding for binding in schedule.flow_routes}
            for flow in dag.flows:
                occurrences.setdefault(flow.id, []).append(
                    (dag.die_id, flow, bindings[flow.id])
                )
            for task in dag.tasks:
                if task.kind is not SemanticTaskKind.BARRIER:
                    continue
                assert task.sync is not None and task.sync.barrier is not None
                barrier = task.sync.barrier
                previous = barrier_definitions.setdefault(barrier.id, barrier)
                if previous != barrier:
                    raise SchemaError(
                        "shared barrier id has conflicting PLAN definitions",
                        path=f"{path}.schedules",
                    )
                barrier_ranks.setdefault(barrier.id, []).append(
                    task.origin_ref.rank
                )
        for barrier_id, barrier in barrier_definitions.items():
            ranks = barrier_ranks[barrier_id]
            if (
                barrier.scope is not BarrierScope.PLAN
                or len(ranks) != len(set(ranks))
                or set(ranks) != set(barrier.participant_ranks)
            ):
                raise SchemaError(
                    f"PLAN barrier {barrier_id!r} requires exactly one action per participant rank",
                    path=f"{path}.schedules",
                )
        route_catalog = _global_route_index(ir1, "ir1")
        for flow_id, entries in occurrences.items():
            canonical = entries[0][1]
            canonical_metadata = (
                canonical.logical_channel,
                canonical.pair_route_ref,
                canonical.source_rank,
                canonical.destination_rank,
                canonical.source_die,
                canonical.destination_die,
                canonical.die_path,
                canonical.tensor_slice,
                canonical.bytes,
                canonical.dtype,
            )
            if any(
                (
                    flow.logical_channel,
                    flow.pair_route_ref,
                    flow.source_rank,
                    flow.destination_rank,
                    flow.source_die,
                    flow.destination_die,
                    flow.die_path,
                    flow.tensor_slice,
                    flow.bytes,
                    flow.dtype,
                )
                != canonical_metadata
                for _die_id, flow, _binding in entries[1:]
            ):
                raise SchemaError(
                    f"logical flow {flow_id!r} has inconsistent cross-die metadata",
                    path=f"{path}.schedules",
                )
            route = route_catalog.get(canonical.pair_route_ref)
            if route is None:
                raise SchemaError(
                    f"logical flow {flow_id!r} references an unknown route",
                    path=f"{path}.schedules",
                )
            roles_by_die = {die_id: binding.role for die_id, _flow, binding in entries}
            expected_roles = {
                die_id: (
                    FlowRouteRole.SOURCE
                    if index == 0
                    else FlowRouteRole.DESTINATION
                    if index == len(route.die_path) - 1
                    else FlowRouteRole.TRANSIT
                )
                for index, die_id in enumerate(route.die_path)
            }
            if len(roles_by_die) != len(entries) or roles_by_die != expected_roles:
                raise SchemaError(
                    f"logical flow {flow_id!r} requires one SOURCE, one DESTINATION, and one TRANSIT per intermediate die",
                    path=f"{path}.schedules",
                )
