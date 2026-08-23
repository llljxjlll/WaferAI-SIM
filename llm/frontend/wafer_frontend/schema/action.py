"""Versioned fusion and standalone-collective action plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .common import (
    DType,
    MeshAxisName,
    ProfileKey,
    RoundingMode,
    TensorValue,
    stable_artifact_id,
    validate_dependency_dag,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .ir0 import (
    AttentionWorkload,
    CollectiveKind,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyBackwardWorkload,
    EdgeKind,
    ElementwiseWorkload,
    EmbeddingWorkload,
    FusionImpl,
    GemmPartition,
    GemmWorkload,
    GreedySampleWorkload,
    NodeEffects,
    NodeMath,
    NodeWorkload,
    NormWorkload,
    OpKind,
    P2PByteWorkload,
    ResidualWorkload,
    ReduceOp,
    RmsNormWorkload,
    RopeQkWorkload,
    SwiGluWorkload,
    SgdUpdateWorkload,
)
from .persistent_state import (
    StateKind,
    canonical_state_staging_value_id,
)

if TYPE_CHECKING:
    from .ir1 import IR1


FUSION_PLAN_SCHEMA_VERSION = "wafer_frontend.fusion_plan/v1alpha10"
STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.standalone_collective_plan/v1alpha9"
)
SWIZZLE_BOUND_ACTION_REF_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_bound_action_ref/v1"
)


class FusionActionKind(str, Enum):
    COMP = "comp"
    LOCAL_COPY = "local_copy"
    SEND = "send"
    RECV = "recv"
    REDUCE = "reduce"
    WAIT = "wait"
    BARRIER = "barrier"


@dataclass(frozen=True, slots=True)
class SwizzleBoundActionRef:
    """Stable common-IR2 reference to one lossless Swizzle bound action.

    ``SwizzleFusionPlan`` deliberately owns richer semantics than the legacy
    ``FusionAction`` carrier (chunk origin, temporary-value lineage and
    deployment selection).  Common IR2 must reference that action directly,
    rather than reconstructing a naive action and losing those semantics.
    The plan itself remains the authoritative payload and is checked by the
    projection boundary.
    """

    plan_id: str
    rank: int
    action_id: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.plan_id, f"{path}.plan_id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.action_id, f"{path}.action_id")


class CollectiveAlgorithm(str, Enum):
    RING = "ring"
    DIRECT = "direct"
    TREE = "tree"


class ChunkDim(str, Enum):
    M = "M"
    N = "N"


class BarrierScope(str, Enum):
    CHUNK = "chunk"
    GROUP = "group"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class ComputeOperand:
    """Ordered rank-local operand and its semantic role."""

    value_id: str
    role: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_id, f"{path}.value_id")
        validate_nonempty(self.role, f"{path}.role")


@dataclass(frozen=True, slots=True)
class ComputeOperandSlice:
    """Map an action-local operand identity to an IR-1 logical tensor slice."""

    operand_id: str
    source_value_id: str
    logical_offset: tuple[int, ...]
    logical_shape: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.operand_id, f"{path}.operand_id")
        validate_nonempty(self.source_value_id, f"{path}.source_value_id")
        if not self.logical_shape or len(self.logical_offset) != len(self.logical_shape):
            raise SchemaError(
                "logical_offset and logical_shape must have equal non-zero rank",
                path=path,
            )
        for field_name in ("logical_offset", "logical_shape"):
            for index, value in enumerate(getattr(self, field_name)):
                validate_uint64(value, f"{path}.{field_name}[{index}]")
                if field_name == "logical_shape" and value == 0:
                    raise SchemaError(
                        "must be greater than zero",
                        path=f"{path}.{field_name}[{index}]",
                    )


@dataclass(frozen=True, slots=True)
class ComputeTileBinding:
    """Self-contained origin semantics and logical slices for one compute tile."""

    origin_workload: GemmWorkload
    input_slices: tuple[ComputeOperandSlice, ...]
    output_slices: tuple[ComputeOperandSlice, ...]

    def validate(self, path: str) -> None:
        self.origin_workload.validate(f"{path}.origin_workload")
        if not self.input_slices or not self.output_slices:
            raise SchemaError(
                "must contain input and output operand slices",
                path=path,
            )
        for field_name in ("input_slices", "output_slices"):
            slices = getattr(self, field_name)
            operand_ids: set[str] = set()
            for index, binding in enumerate(slices):
                binding.validate(f"{path}.{field_name}[{index}]")
                if binding.operand_id in operand_ids:
                    raise SchemaError(
                        "contains a duplicate operand_id",
                        path=f"{path}.{field_name}[{index}].operand_id",
                    )
                operand_ids.add(binding.operand_id)


def _validate_workload_kind(
    op_kind: OpKind,
    workload: NodeWorkload,
    *,
    path: str,
) -> None:
    if type(op_kind) is not OpKind:
        raise SchemaError("must be an OpKind", path=f"{path}.op_kind")
    expected_types = {
        OpKind.GEMM: (GemmWorkload,),
        OpKind.ATTENTION: (AttentionWorkload,),
        OpKind.COLLECTIVE: (CollectiveWorkload,),
        OpKind.ELEMENTWISE: (
            ElementwiseWorkload,
            SwiGluWorkload,
            ResidualWorkload,
        ),
        OpKind.NORM: (NormWorkload, RmsNormWorkload),
        OpKind.P2P: (P2PByteWorkload,),
        OpKind.EMBEDDING: (EmbeddingWorkload,),
        OpKind.ROPE: (RopeQkWorkload,),
        OpKind.SAMPLING: (GreedySampleWorkload,),
        OpKind.CE_FORWARD: (CrossEntropyForwardWorkload,),
        OpKind.CE_BACKWARD: (CrossEntropyBackwardWorkload,),
        OpKind.OPTIMIZER_UPDATE: (SgdUpdateWorkload,),
    }.get(op_kind)
    if expected_types is None:
        raise SchemaError(
            f"op_kind {op_kind.value!r} has no compute carrier",
            path=f"{path}.op_kind",
        )
    if type(workload) not in expected_types:
        raise SchemaError(
            f"workload type must match op_kind {op_kind.value!r}",
            path=f"{path}.workload",
        )


def canonical_compute_operand_roles(
    op_kind: OpKind,
    workload: NodeWorkload,
    *,
    tiled: bool,
    path: str = "compute",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the only legal ordered operand-role contract for one compute."""

    _validate_workload_kind(op_kind, workload, path=path)
    if type(tiled) is not bool:
        raise SchemaError("must be a bool", path=f"{path}.tile")
    if tiled and op_kind is not OpKind.GEMM:
        raise SchemaError(
            "compute tile binding is supported only for GEMM",
            path=f"{path}.tile",
        )
    if op_kind is OpKind.GEMM:
        return ("lhs", "rhs"), (("partial",) if tiled else ("output",))
    if op_kind is OpKind.EMBEDDING:
        return ("indices", "table"), ("activation",)
    if type(workload) is RmsNormWorkload:
        return ("activation", "weight"), ("normalized",)
    if op_kind is OpKind.ROPE:
        return ("packed_qkv",), ("packed_qkv",)
    if op_kind is OpKind.ATTENTION:
        return ("packed_qkv",), ("attention_output",)
    if type(workload) is SwiGluWorkload:
        return ("gate_up",), ("swiglu",)
    if type(workload) is ResidualWorkload:
        return ("residual", "branch"), ("output",)
    if op_kind is OpKind.SAMPLING:
        return ("logits",), ("sample_ids",)
    if op_kind is OpKind.CE_FORWARD:
        return ("logits", "labels"), ("loss",)
    if op_kind is OpKind.CE_BACKWARD:
        return (
            "logits",
            "labels",
            "loss_gradient",
        ), ("logits_gradient",)
    if op_kind is OpKind.OPTIMIZER_UPDATE:
        return ("weight", "weight_gradient"), ("updated_weight",)

    # Legacy S1 carriers retain their already-published ordered role spelling.
    if type(workload) is NormWorkload:
        return ("input_0",), ("output_0",)
    if type(workload) is ElementwiseWorkload:
        return (
            tuple(
                f"input_{index}"
                for index in range(len(workload.rank_input_shapes))
            ),
            ("output_0",),
        )
    if op_kind is OpKind.P2P:
        return ("input_0",), ("output_0",)
    if op_kind is OpKind.COLLECTIVE:
        return ("input_0",), ("output_0",)
    raise SchemaError("compute operand contract is not implemented", path=path)


@dataclass(frozen=True, slots=True)
class ComputeContract:
    """Self-contained rank-local compute semantics; no IR-1 lookup is needed later."""

    op_kind: OpKind
    workload: NodeWorkload
    math: NodeMath
    effects: NodeEffects
    impl_ref: str
    inputs: tuple[ComputeOperand, ...]
    outputs: tuple[ComputeOperand, ...]
    tile: ComputeTileBinding | None = None

    def validate(self, path: str) -> None:
        _validate_workload_kind(self.op_kind, self.workload, path=path)
        self.workload.validate(f"{path}.workload")
        self.math.validate(f"{path}.math")
        self.effects.validate(f"{path}.effects")
        validate_nonempty(self.impl_ref, f"{path}.impl_ref")
        if not self.outputs:
            raise SchemaError("must contain at least one output", path=f"{path}.outputs")
        for field_name in ("inputs", "outputs"):
            operands = getattr(self, field_name)
            ids: set[str] = set()
            for index, operand in enumerate(operands):
                operand.validate(f"{path}.{field_name}[{index}]")
                if operand.value_id in ids:
                    raise SchemaError(
                        "contains duplicate value ids",
                        path=f"{path}.{field_name}[{index}].value_id",
                    )
                ids.add(operand.value_id)
        expected_input_roles, expected_output_roles = canonical_compute_operand_roles(
            self.op_kind,
            self.workload,
            tiled=self.tile is not None,
            path=path,
        )
        for field_name, expected_roles in (
            ("inputs", expected_input_roles),
            ("outputs", expected_output_roles),
        ):
            actual_roles = tuple(
                operand.role for operand in getattr(self, field_name)
            )
            if actual_roles != expected_roles:
                raise SchemaError(
                    f"operand roles/arity must exactly equal {expected_roles!r}",
                    path=f"{path}.{field_name}",
                )
        if self.tile is not None:
            if self.op_kind is not OpKind.GEMM or not isinstance(
                self.workload, GemmWorkload
            ):
                raise SchemaError(
                    "compute tile binding is supported only for GEMM",
                    path=f"{path}.tile",
                )
            self.tile.validate(f"{path}.tile")
            if tuple(item.operand_id for item in self.tile.input_slices) != tuple(
                item.value_id for item in self.inputs
            ):
                raise SchemaError(
                    "tile input operand order must exactly match compute inputs",
                    path=f"{path}.tile.input_slices",
                )
            if tuple(item.operand_id for item in self.tile.output_slices) != tuple(
                item.value_id for item in self.outputs
            ):
                raise SchemaError(
                    "tile output operand order must exactly match compute outputs",
                    path=f"{path}.tile.output_slices",
                )


@dataclass(frozen=True, slots=True)
class BarrierContract:
    id: str
    participant_ranks: tuple[int, ...]
    arrival_count: int
    scope: BarrierScope

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        if not self.participant_ranks:
            raise SchemaError(
                "must contain participant ranks", path=f"{path}.participant_ranks"
            )
        if len(set(self.participant_ranks)) != len(self.participant_ranks):
            raise SchemaError(
                "contains duplicate ranks", path=f"{path}.participant_ranks"
            )
        for index, rank in enumerate(self.participant_ranks):
            validate_uint64(rank, f"{path}.participant_ranks[{index}]")
        validate_uint64(self.arrival_count, f"{path}.arrival_count")
        if self.arrival_count != len(self.participant_ranks):
            raise SchemaError(
                "must equal participant_ranks length", path=f"{path}.arrival_count"
            )


@dataclass(frozen=True, slots=True)
class SyncContract:
    """Logical synchronization only; the finalizer assigns physical event/token ids."""

    completion_event: str
    wait_event: str | None
    barrier: BarrierContract | None

    def validate_for_kind(self, kind: FusionActionKind, path: str) -> None:
        validate_nonempty(self.completion_event, f"{path}.completion_event")
        if self.wait_event is not None:
            validate_nonempty(self.wait_event, f"{path}.wait_event")
        if self.barrier is not None:
            self.barrier.validate(f"{path}.barrier")
        if kind is FusionActionKind.WAIT:
            if self.wait_event is None or self.barrier is not None:
                raise SchemaError(
                    "WAIT requires wait_event and forbids barrier", path=path
                )
        elif kind is FusionActionKind.BARRIER:
            if self.barrier is None or self.wait_event is not None:
                raise SchemaError(
                    "BARRIER requires barrier and forbids wait_event", path=path
                )
        elif self.wait_event is not None or self.barrier is not None:
            raise SchemaError(
                "only WAIT/BARRIER may carry wait or barrier bindings", path=path
            )


@dataclass(frozen=True, slots=True)
class ReductionContract:
    reduce_op: ReduceOp
    input_dtype: DType
    accumulation_dtype: DType
    output_dtype: DType
    rounding: RoundingMode
    input_ranks: tuple[int, ...]

    def validate(self, path: str) -> None:
        legacy = (
            self.reduce_op is ReduceOp.SUM
            and self.input_dtype is DType.FP16
            and self.accumulation_dtype is DType.FP32
            and self.output_dtype is DType.FP16
            and self.rounding is RoundingMode.RNE
        )
        dp2_fp32 = (
            self.reduce_op is ReduceOp.SUM
            and self.input_dtype is DType.FP32
            and self.accumulation_dtype is DType.FP32
            and self.output_dtype is DType.FP32
            and self.rounding is RoundingMode.RNE
            and self.input_ranks == (0, 1)
        )
        if not (legacy or dp2_fp32):
            raise SchemaError(
                "naive backend v1 requires SUM FP16->FP32->FP16 with RNE or exact DP2 FP32->FP32->FP32 with ranks (0, 1)",
                path=path,
            )
        if not self.input_ranks:
            raise SchemaError("must contain at least one input rank", path=f"{path}.input_ranks")
        if len(set(self.input_ranks)) != len(self.input_ranks):
            raise SchemaError("contains duplicate ranks", path=f"{path}.input_ranks")
        for index, rank in enumerate(self.input_ranks):
            validate_uint64(rank, f"{path}.input_ranks[{index}]")


@dataclass(frozen=True, slots=True)
class ChunkSlice:
    id: str
    chunk_id: int
    value_id: str
    offset: tuple[int, ...]
    shape: tuple[int, ...]
    bytes: int
    owner_rank: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.chunk_id, f"{path}.chunk_id")
        validate_nonempty(self.value_id, f"{path}.value_id")
        if not self.shape or len(self.offset) != len(self.shape):
            raise SchemaError(
                "offset and shape must have the same non-zero rank", path=path
            )
        for field_name in ("offset", "shape"):
            for index, value in enumerate(getattr(self, field_name)):
                validate_uint64(value, f"{path}.{field_name}[{index}]")
                if field_name == "shape" and value == 0:
                    raise SchemaError(
                        "must be greater than zero", path=f"{path}.shape[{index}]"
                    )
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        validate_uint64(self.owner_rank, f"{path}.owner_rank")


@dataclass(frozen=True, slots=True)
class FusionAction:
    id: str
    kind: FusionActionKind
    member_id: str | None
    chunk_id: int | None
    collective_step: int | None
    peer_rank: int | None
    expected_route: tuple[int, ...]
    slice_ref: str | None
    bytes: int
    dtype: DType | None
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    logical_channel: str | None
    compute: ComputeContract | None
    reduction: ReductionContract | None
    sync: SyncContract
    deps: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        for field_name in ("member_id", "slice_ref", "logical_channel"):
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty(value, f"{path}.{field_name}")
        for field_name in ("chunk_id", "collective_step", "peer_rank"):
            value = getattr(self, field_name)
            if value is not None:
                validate_uint64(value, f"{path}.{field_name}")
        validate_uint64(self.bytes, f"{path}.bytes")
        if self.compute is not None:
            self.compute.validate(f"{path}.compute")
        if self.reduction is not None:
            self.reduction.validate(f"{path}.reduction")
        self.sync.validate_for_kind(self.kind, f"{path}.sync")
        for index, die_id in enumerate(self.expected_route):
            validate_uint64(die_id, f"{path}.expected_route[{index}]")
        for field_name in ("reads", "writes"):
            refs = getattr(self, field_name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate value ids", path=f"{path}.{field_name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{field_name}[{index}]")

        if self.kind in (FusionActionKind.SEND, FusionActionKind.RECV):
            if self.peer_rank is None:
                raise SchemaError("is required for SEND/RECV", path=f"{path}.peer_rank")
            if len(self.expected_route) < 2:
                raise SchemaError(
                    "must contain source and destination dies",
                    path=f"{path}.expected_route",
                )
            if self.slice_ref is None or self.chunk_id is None:
                raise SchemaError("SEND/RECV requires chunk and slice", path=path)
            if self.bytes == 0 or self.dtype is None:
                raise SchemaError("SEND/RECV requires non-zero bytes and dtype", path=path)
            if self.logical_channel is None:
                raise SchemaError("is required for SEND/RECV", path=f"{path}.logical_channel")
            if self.kind is FusionActionKind.SEND and (not self.reads or self.writes):
                raise SchemaError("SEND must read payload and cannot write values", path=path)
            if self.kind is FusionActionKind.RECV and (self.reads or not self.writes):
                raise SchemaError("RECV must write payload and cannot read values", path=path)
        elif self.kind in (FusionActionKind.WAIT, FusionActionKind.BARRIER):
            if any(
                value is not None
                for value in (self.peer_rank, self.slice_ref, self.logical_channel, self.dtype)
            ) or self.expected_route or self.bytes != 0:
                raise SchemaError("WAIT/BARRIER cannot carry transport payload", path=path)
            if self.reads or self.writes:
                raise SchemaError("WAIT/BARRIER cannot read or write values", path=path)
        else:
            if self.peer_rank is not None or self.expected_route or self.logical_channel is not None:
                raise SchemaError("non-transport action cannot carry peer/route/channel", path=path)
            if self.slice_ref is None or self.chunk_id is None:
                raise SchemaError("compute/reduce action requires chunk and slice", path=path)
            if self.bytes == 0 or self.dtype is None:
                raise SchemaError("compute/reduce action requires non-zero bytes and dtype", path=path)
        if self.kind is FusionActionKind.REDUCE:
            if self.reduction is None:
                raise SchemaError("is required for REDUCE", path=f"{path}.reduction")
            if self.dtype is not self.reduction.input_dtype:
                raise SchemaError("dtype must equal reduction input_dtype", path=f"{path}.dtype")
        elif self.reduction is not None:
            raise SchemaError("is only valid for REDUCE", path=f"{path}.reduction")
        if self.kind is FusionActionKind.COMP:
            if self.compute is None:
                raise SchemaError("is required for COMP", path=f"{path}.compute")
            if self.member_id is None:
                raise SchemaError("is required for COMP", path=f"{path}.member_id")
            if tuple(item.value_id for item in self.compute.inputs) != self.reads:
                raise SchemaError("inputs must exactly match reads", path=f"{path}.compute.inputs")
            if tuple(item.value_id for item in self.compute.outputs) != self.writes:
                raise SchemaError("outputs must exactly match writes", path=f"{path}.compute.outputs")
        elif self.compute is not None:
            raise SchemaError("is only valid for COMP", path=f"{path}.compute")
        if self.kind is FusionActionKind.LOCAL_COPY:
            if self.member_id is None or len(self.reads) != 1 or len(self.writes) != 1:
                raise SchemaError(
                    "LOCAL_COPY requires exactly one input and one output", path=path
                )


@dataclass(frozen=True, slots=True)
class RankProgram:
    rank: int
    actions: tuple[FusionAction, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if not self.actions:
            raise SchemaError("must contain at least one action", path=f"{path}.actions")
        validate_dependency_dag(self.actions, f"{path}.actions")


@dataclass(frozen=True, slots=True)
class PermutationEntry:
    chunk_id: int
    logical_owner_rank: int
    physical_owner_rank: int

    def validate(self, path: str) -> None:
        for field_name in ("chunk_id", "logical_owner_rank", "physical_owner_rank"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class InversePermutationEntry:
    chunk_id: int
    physical_owner_rank: int
    logical_owner_rank: int

    def validate(self, path: str) -> None:
        for field_name in ("chunk_id", "physical_owner_rank", "logical_owner_rank"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")


@dataclass(frozen=True, slots=True)
class ConsumerLayoutBinding:
    consumer_node_id: str
    value_id: str
    accepted_layout: str
    applies_inverse_permutation: bool

    def validate(self, path: str) -> None:
        for field_name in ("consumer_node_id", "value_id", "accepted_layout"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")


def _validate_rank_programs(
    programs: tuple[RankProgram, ...],
    chunks: tuple[ChunkSlice, ...],
    *,
    path: str,
) -> None:
    if not programs:
        raise SchemaError("must contain rank programs", path=f"{path}.rank_programs")
    ranks: set[int] = set()
    action_ids: set[str] = set()
    completion_events: dict[str, FusionAction] = {}
    actions_by_channel: dict[str, list[tuple[int, FusionAction]]] = {}
    barriers: dict[str, tuple[BarrierContract, set[int]]] = {}
    chunk_index = validate_unique_ids(chunks, f"{path}.chunk_slices")
    chunks_by_number = {chunk.chunk_id: chunk for chunk in chunks}
    for index, chunk in enumerate(chunks):
        chunk.validate(f"{path}.chunk_slices[{index}]")
        if chunk.chunk_id in {item.chunk_id for item in chunks[:index]}:
            raise SchemaError("duplicate chunk_id", path=f"{path}.chunk_slices[{index}].chunk_id")
    for program_index, program in enumerate(programs):
        program.validate(f"{path}.rank_programs[{program_index}]")
        if program.rank in ranks:
            raise SchemaError("duplicate rank", path=f"{path}.rank_programs[{program_index}].rank")
        ranks.add(program.rank)
        for action_index, action in enumerate(program.actions):
            action.validate(f"{path}.rank_programs[{program_index}].actions[{action_index}]")
            if action.id in action_ids:
                raise SchemaError("action ids must be plan-global", path=f"{path}.rank_programs[{program_index}].actions[{action_index}].id")
            action_ids.add(action.id)
            if action.sync.completion_event in completion_events:
                raise SchemaError(
                    "completion events must be plan-global unique",
                    path=f"{path}.rank_programs[{program_index}].actions[{action_index}].sync.completion_event",
                )
            completion_events[action.sync.completion_event] = action
            if action.slice_ref is not None:
                chunk = chunk_index.get(action.slice_ref)
                if chunk is None:
                    raise SchemaError("dangling chunk slice", path=f"{path}.rank_programs[{program_index}].actions[{action_index}].slice_ref")
                if action.chunk_id != chunk.chunk_id or action.bytes != chunk.bytes:
                    raise SchemaError("action payload disagrees with chunk slice", path=f"{path}.rank_programs[{program_index}].actions[{action_index}]")
            if action.chunk_id is not None and action.chunk_id not in chunks_by_number:
                raise SchemaError("dangling chunk_id", path=f"{path}.rank_programs[{program_index}].actions[{action_index}].chunk_id")
            if action.peer_rank is not None and action.peer_rank not in range(len(programs)):
                raise SchemaError("dangling peer rank", path=f"{path}.rank_programs[{program_index}].actions[{action_index}].peer_rank")
            if action.logical_channel is not None:
                actions_by_channel.setdefault(action.logical_channel, []).append((program.rank, action))
            if action.sync.barrier is not None:
                prior = barriers.get(action.sync.barrier.id)
                if prior is None:
                    barriers[action.sync.barrier.id] = (action.sync.barrier, {program.rank})
                else:
                    contract, participant_actions = prior
                    if contract != action.sync.barrier:
                        raise SchemaError(
                            "barrier id has conflicting definitions",
                            path=f"{path}.rank_programs[{program_index}].actions[{action_index}].sync.barrier",
                        )
                    participant_actions.add(program.rank)
    if ranks != set(range(len(programs))):
        raise SchemaError("ranks must be contiguous from zero", path=f"{path}.rank_programs")
    for index, chunk in enumerate(chunks):
        if chunk.owner_rank not in ranks:
            raise SchemaError("chunk owner is not a program rank", path=f"{path}.chunk_slices[{index}].owner_rank")
    for program_index, program in enumerate(programs):
        for action_index, action in enumerate(program.actions):
            if action.reduction is not None and not set(action.reduction.input_ranks).issubset(ranks):
                raise SchemaError(
                    "reduction input_ranks contain a non-program rank",
                    path=f"{path}.rank_programs[{program_index}].actions[{action_index}].reduction.input_ranks",
                )
            if action.kind is FusionActionKind.WAIT:
                waited = completion_events.get(action.sync.wait_event or "")
                if waited is None:
                    raise SchemaError(
                        "WAIT references an unknown completion event",
                        path=f"{path}.rank_programs[{program_index}].actions[{action_index}].sync.wait_event",
                    )
                if waited.id not in action.deps:
                    raise SchemaError(
                        "WAIT dependency must name the waited producer action",
                        path=f"{path}.rank_programs[{program_index}].actions[{action_index}].deps",
                    )
            barrier = action.sync.barrier
            if barrier is not None:
                if program.rank not in barrier.participant_ranks:
                    raise SchemaError(
                        "barrier action rank is not a participant",
                        path=f"{path}.rank_programs[{program_index}].actions[{action_index}].sync.barrier.participant_ranks",
                    )
                if not set(barrier.participant_ranks).issubset(ranks):
                    raise SchemaError(
                        "barrier contains a non-program rank",
                        path=f"{path}.rank_programs[{program_index}].actions[{action_index}].sync.barrier.participant_ranks",
                    )
    for barrier_id, (barrier, participant_actions) in barriers.items():
        if participant_actions != set(barrier.participant_ranks):
            raise SchemaError(
                f"barrier {barrier_id!r} must have exactly one action per participant",
                path=f"{path}.rank_programs",
            )
    for channel, endpoints in actions_by_channel.items():
        if len(endpoints) != 2:
            raise SchemaError(f"logical channel {channel!r} must have one SEND and one RECV", path=f"{path}.rank_programs")
        send_endpoints = [item for item in endpoints if item[1].kind is FusionActionKind.SEND]
        recv_endpoints = [item for item in endpoints if item[1].kind is FusionActionKind.RECV]
        if len(send_endpoints) != 1 or len(recv_endpoints) != 1:
            raise SchemaError(f"logical channel {channel!r} must pair SEND with RECV", path=f"{path}.rank_programs")
        send_rank, send = send_endpoints[0]
        recv_rank, recv = recv_endpoints[0]
        if send.peer_rank != recv_rank or recv.peer_rank != send_rank:
            raise SchemaError(f"logical channel {channel!r} has non-reciprocal peers", path=f"{path}.rank_programs")
        if (
            send.expected_route,
            send.slice_ref,
            send.chunk_id,
            send.collective_step,
            send.bytes,
            send.dtype,
        ) != (
            recv.expected_route,
            recv.slice_ref,
            recv.chunk_id,
            recv.collective_step,
            recv.bytes,
            recv.dtype,
        ):
            raise SchemaError(f"logical channel {channel!r} payloads do not match", path=f"{path}.rank_programs")


def _actions_by_rank_and_chunk(
    programs: tuple[RankProgram, ...],
) -> dict[tuple[int, int, FusionActionKind], list[FusionAction]]:
    result: dict[tuple[int, int, FusionActionKind], list[FusionAction]] = {}
    for program in programs:
        for action in program.actions:
            if action.chunk_id is not None:
                result.setdefault((program.rank, action.chunk_id, action.kind), []).append(action)
    return result


def _require_one(
    index: dict[tuple[int, int, FusionActionKind], list[FusionAction]],
    rank: int,
    chunk_id: int,
    kind: FusionActionKind,
    *,
    path: str,
) -> FusionAction:
    actions = index.get((rank, chunk_id, kind), ())
    if len(actions) != 1:
        raise SchemaError(
            f"DIRECT requires exactly one {kind.value} for rank {rank}, chunk {chunk_id}",
            path=f"{path}.rank_programs",
        )
    return actions[0]


def _validate_direct_fusion_execution(
    programs: tuple[RankProgram, ...], chunks: tuple[ChunkSlice, ...], *, path: str
) -> None:
    """Freeze the canonical direct naive fused GEMM+RS program shape."""

    index = _actions_by_rank_and_chunk(programs)
    ranks = tuple(sorted(program.rank for program in programs))
    allowed = {
        FusionActionKind.COMP,
        FusionActionKind.SEND,
        FusionActionKind.RECV,
        FusionActionKind.WAIT,
        FusionActionKind.REDUCE,
    }
    if any(action.kind not in allowed for program in programs for action in program.actions):
        raise SchemaError("DIRECT fused plan contains an unsupported action kind", path=f"{path}.rank_programs")
    expected_actions: dict[int, list[FusionAction]] = {rank: [] for rank in ranks}
    contribution_ids: set[str] = set()
    for chunk in chunks:
        owner = chunk.owner_rank
        comps = {
            rank: _require_one(index, rank, chunk.chunk_id, FusionActionKind.COMP, path=path)
            for rank in ranks
        }
        for rank, comp in comps.items():
            if (
                len(comp.writes) != 1
                or comp.deps
                or comp.collective_step is not None
            ):
                raise SchemaError(
                    "each COMP must be an independent step-less single contribution",
                    path=f"{path}.rank_programs",
                )
            contribution_id = comp.writes[0]
            if contribution_id in contribution_ids:
                raise SchemaError(
                    "contribution temporaries must be plan-global unique",
                    path=f"{path}.rank_programs",
                )
            contribution_ids.add(contribution_id)
            expected_actions[rank].append(comp)
        reduce = _require_one(index, owner, chunk.chunk_id, FusionActionKind.REDUCE, path=path)
        if (
            reduce.reduction is None
            or reduce.reduction.input_ranks != ranks
            or reduce.collective_step != 1
        ):
            raise SchemaError(
                "REDUCE must be step 1 with input_ranks exactly in group rank order",
                path=f"{path}.rank_programs",
            )
        if reduce.writes != (chunk.value_id,):
            raise SchemaError(
                "the unique chunk REDUCE must write its boundary value",
                path=f"{path}.rank_programs",
            )
        recv_by_peer: dict[int, FusionAction] = {}
        wait_by_peer: dict[int, FusionAction] = {}
        for rank in ranks:
            sends = index.get((rank, chunk.chunk_id, FusionActionKind.SEND), ())
            recvs = index.get((rank, chunk.chunk_id, FusionActionKind.RECV), ())
            waits = index.get((rank, chunk.chunk_id, FusionActionKind.WAIT), ())
            reduces = index.get((rank, chunk.chunk_id, FusionActionKind.REDUCE), ())
            if rank == owner:
                if (
                    sends
                    or len(recvs) != len(ranks) - 1
                    or len(waits) != len(ranks) - 1
                    or len(reduces) != 1
                ):
                    raise SchemaError("owner communication/reduction shape is not DIRECT", path=f"{path}.rank_programs")
                for recv in recvs:
                    if (
                        recv.peer_rank == owner
                        or recv.collective_step != 0
                        or recv.deps
                    ):
                        raise SchemaError("owner cannot receive from itself", path=f"{path}.rank_programs")
                    if recv.peer_rank in recv_by_peer or len(recv.writes) != 1:
                        raise SchemaError(
                            "owner requires one single-write RECV per peer",
                            path=f"{path}.rank_programs",
                        )
                    recv_by_peer[recv.peer_rank] = recv
                    contribution_id = recv.writes[0]
                    if contribution_id in contribution_ids:
                        raise SchemaError(
                            "contribution temporaries must be plan-global unique",
                            path=f"{path}.rank_programs",
                        )
                    contribution_ids.add(contribution_id)
                for wait in waits:
                    matching_peers = tuple(
                        peer
                        for peer, recv in recv_by_peer.items()
                        if wait.sync.wait_event == recv.sync.completion_event
                    )
                    if (
                        len(matching_peers) != 1
                        or wait.collective_step != 0
                        or wait.deps != (recv_by_peer[matching_peers[0]].id,)
                    ):
                        raise SchemaError(
                            "each owner WAIT must wait on exactly one same-rank RECV",
                            path=f"{path}.rank_programs",
                        )
                    peer = matching_peers[0]
                    if peer in wait_by_peer:
                        raise SchemaError(
                            "owner requires one WAIT per peer RECV",
                            path=f"{path}.rank_programs",
                        )
                    wait_by_peer[peer] = wait
            else:
                send = _require_one(index, rank, chunk.chunk_id, FusionActionKind.SEND, path=path)
                if (
                    send.peer_rank != owner
                    or send.collective_step != 0
                    or send.deps != (comps[rank].id,)
                    or send.reads != comps[rank].writes
                ):
                    raise SchemaError(
                        "non-owner SEND must read and depend on its local COMP contribution",
                        path=f"{path}.rank_programs",
                    )
                if recvs or waits or reduces:
                    raise SchemaError("non-owner cannot RECV, WAIT, or REDUCE this chunk", path=f"{path}.rank_programs")
                expected_actions[rank].append(send)
        if set(recv_by_peer) != set(ranks) - {owner} or set(wait_by_peer) != set(ranks) - {owner}:
            raise SchemaError(
                "owner requires one RECV and WAIT for every peer",
                path=f"{path}.rank_programs",
            )
        for peer in ranks:
            if peer != owner:
                expected_actions[owner].extend((recv_by_peer[peer], wait_by_peer[peer]))
        expected_reduce_deps = tuple(
            comps[rank].id if rank == owner else wait_by_peer[rank].id
            for rank in ranks
        )
        if reduce.deps != expected_reduce_deps:
            raise SchemaError("REDUCE deps must follow input_ranks using local COMP or RECV WAIT", path=f"{path}.rank_programs")
        assert reduce.reduction is not None
        expected_reads = tuple(
            comps[rank].writes[0]
            if rank == owner
            else recv_by_peer[rank].writes[0]
            for rank in reduce.reduction.input_ranks
        )
        if reduce.reads != expected_reads:
            raise SchemaError(
                "REDUCE reads must follow input_ranks and exact contribution lineage",
                path=f"{path}.rank_programs",
            )
        expected_actions[owner].append(reduce)
    for program in programs:
        if program.actions != tuple(expected_actions[program.rank]):
            raise SchemaError(
                "DIRECT fused actions must be in canonical chunk/kind/peer order",
                path=f"{path}.rank_programs[{program.rank}].actions",
            )


def _validate_direct_all_gather_execution(
    programs: tuple[RankProgram, ...], chunks: tuple[ChunkSlice, ...], *, path: str
) -> None:
    """Standalone v1 supports explicit local placement plus direct AllGather only."""

    index = _actions_by_rank_and_chunk(programs)
    ranks = tuple(sorted(program.rank for program in programs))
    allowed = {FusionActionKind.LOCAL_COPY, FusionActionKind.SEND, FusionActionKind.RECV, FusionActionKind.BARRIER}
    if any(action.kind not in allowed for program in programs for action in program.actions):
        raise SchemaError("DIRECT AllGather contains an unsupported action kind", path=f"{path}.rank_programs")
    local_completion: dict[int, set[str]] = {rank: set() for rank in ranks}
    expected_actions: dict[int, list[FusionAction]] = {rank: [] for rank in ranks}
    for chunk in chunks:
        owner = chunk.owner_rank
        local = _require_one(index, owner, chunk.chunk_id, FusionActionKind.LOCAL_COPY, path=path)
        if local.collective_step != 0 or local.deps:
            raise SchemaError(
                "owner LOCAL_COPY must be an independent step 0 action",
                path=f"{path}.rank_programs",
            )
        local_completion[owner].add(local.id)
        expected_actions[owner].append(local)
        for rank in ranks:
            copies = index.get((rank, chunk.chunk_id, FusionActionKind.LOCAL_COPY), ())
            if rank != owner and copies:
                raise SchemaError("only the owner may place the local contribution", path=f"{path}.rank_programs")
            if rank == owner:
                sends = index.get((rank, chunk.chunk_id, FusionActionKind.SEND), ())
                recvs = index.get((rank, chunk.chunk_id, FusionActionKind.RECV), ())
                if recvs or len(sends) != len(ranks) - 1 or any(
                    send.deps != (local.id,) or send.collective_step != 0
                    for send in sends
                ):
                    raise SchemaError("owner must SEND the local contribution to every peer", path=f"{path}.rank_programs")
                sends_by_peer = {send.peer_rank: send for send in sends}
                if set(sends_by_peer) != set(ranks) - {owner}:
                    raise SchemaError("owner requires one SEND per peer", path=f"{path}.rank_programs")
                expected_actions[owner].extend(sends_by_peer[peer] for peer in ranks if peer != owner)
            else:
                if index.get((rank, chunk.chunk_id, FusionActionKind.SEND), ()):
                    raise SchemaError("non-owner cannot SEND this chunk", path=f"{path}.rank_programs")
                send = [
                    candidate
                    for candidate in index.get((owner, chunk.chunk_id, FusionActionKind.SEND), ())
                    if candidate.peer_rank == rank
                ]
                recv = [
                    candidate
                    for candidate in index.get((rank, chunk.chunk_id, FusionActionKind.RECV), ())
                    if candidate.peer_rank == owner
                ]
                if len(send) != 1 or len(recv) != 1:
                    raise SchemaError("each non-owner requires one owner SEND/peer RECV pair", path=f"{path}.rank_programs")
                if recv[0].collective_step != 0 or recv[0].deps:
                    raise SchemaError(
                        "AllGather RECV must be an independent step 0 action",
                        path=f"{path}.rank_programs",
                    )
                local_completion[rank].add(recv[0].id)
                expected_actions[rank].append(recv[0])
    barrier_defs: set[BarrierContract] = set()
    for program in programs:
        barriers = [action for action in program.actions if action.kind is FusionActionKind.BARRIER]
        if len(barriers) != 1:
            raise SchemaError("DIRECT AllGather requires one completion barrier per rank", path=f"{path}.rank_programs")
        barrier = barriers[0]
        if barrier.sync.barrier is None or barrier.sync.barrier.scope is not BarrierScope.PLAN:
            raise SchemaError("AllGather completion barrier must have plan scope", path=f"{path}.rank_programs")
        if set(barrier.deps) != local_completion[program.rank]:
            raise SchemaError("completion barrier must depend on every locally placed chunk", path=f"{path}.rank_programs")
        expected_barrier_deps = tuple(
            next(
                action.id
                for action in expected_actions[program.rank]
                if action.chunk_id == chunk.chunk_id
                and action.kind in (FusionActionKind.LOCAL_COPY, FusionActionKind.RECV)
            )
            for chunk in chunks
        )
        if barrier.deps != expected_barrier_deps or barrier.collective_step is not None:
            raise SchemaError(
                "completion barrier deps must follow canonical chunk order",
                path=f"{path}.rank_programs",
            )
        barrier_defs.add(barrier.sync.barrier)
        expected_actions[program.rank].append(barrier)
    if len(barrier_defs) != 1:
        raise SchemaError("all ranks must share one completion barrier definition", path=f"{path}.rank_programs")
    for program in programs:
        if program.actions != tuple(expected_actions[program.rank]):
            raise SchemaError(
                "DIRECT AllGather actions must be in canonical chunk/kind/peer order",
                path=f"{path}.rank_programs[{program.rank}].actions",
            )


def _validate_chunk_cover(
    chunks: tuple[ChunkSlice, ...], value: TensorValue, chunk_dim: ChunkDim, *, path: str
) -> None:
    axis = 0 if chunk_dim is ChunkDim.M else 1
    if len(value.shape) <= axis:
        raise SchemaError("chunk dimension is outside the value rank", path=f"{path}.chunk_dim")
    intervals: list[tuple[int, int]] = []
    for index, chunk in enumerate(chunks):
        if chunk.value_id != value.id:
            raise SchemaError("all chunks must slice the same boundary value", path=f"{path}.chunk_slices[{index}].value_id")
        for other_axis, extent in enumerate(value.shape):
            if other_axis != axis and (chunk.offset[other_axis] != 0 or chunk.shape[other_axis] != extent):
                raise SchemaError("chunks must cover every non-chunk dimension", path=f"{path}.chunk_slices[{index}]")
        intervals.append((chunk.offset[axis], chunk.offset[axis] + chunk.shape[axis]))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise SchemaError("chunk slices must be contiguous and disjoint", path=f"{path}.chunk_slices")
        cursor = end
    if cursor != value.shape[axis]:
        raise SchemaError("chunk slices must fully cover the boundary value", path=f"{path}.chunk_slices")


def _owner_profile(ir1: "IR1", instance_id: str, *, path: str) -> ProfileKey:
    """Resolve the one profile that owns a physical Stage 4 instance."""

    if not ir1.instance_profiles:
        if len(ir1.instances) != 1 or ir1.instances[0].id != instance_id:
            raise SchemaError(
                "legacy profile requires the sole IR-1 instance",
                path=path,
            )
        return ir1.profile
    matches = tuple(
        binding.profile
        for binding in ir1.instance_profiles
        if binding.instance_ref == instance_id
    )
    if len(matches) != 1:
        raise SchemaError(
            "owner instance must have exactly one profile binding",
            path=path,
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class FusionPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    fused_op_id: str
    group_ref: str
    impl: FusionImpl
    profile_key: ProfileKey
    collective_algorithm: CollectiveAlgorithm
    chunk_dim: ChunkDim
    chunk_count: int
    chunk_slices: tuple[ChunkSlice, ...]
    rank_programs: tuple[RankProgram, ...]
    input_layout: str
    logical_output_layout: str
    physical_output_layout: str
    output_permutation: tuple[PermutationEntry, ...]
    inverse_permutation: tuple[InversePermutationEntry, ...]
    consumer_layout_bindings: tuple[ConsumerLayoutBinding, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "FusionPlan":
        return cls(
            schema_version=FUSION_PLAN_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("fusion_plan", semantic_key, schema_version=FUSION_PLAN_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "fused_op_id", "group_ref", "impl", "profile_key",
            "collective_algorithm", "chunk_dim", "chunk_count", "chunk_slices",
            "rank_programs", "input_layout", "logical_output_layout",
            "physical_output_layout", "output_permutation", "inverse_permutation",
            "consumer_layout_bindings",
        )}

    def validate(self, path: str = "fusion_plan") -> None:
        if self.schema_version != FUSION_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for field_name in ("producer_pass", "source_ir1_id", "fused_op_id", "group_ref", "input_layout", "logical_output_layout", "physical_output_layout"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if self.impl is not FusionImpl.NAIVE:
            raise SchemaError(
                "DIRECT N4a plan requires impl=naive",
                path=f"{path}.impl",
            )
        self.profile_key.validate(f"{path}.profile_key")
        validate_uint64(self.chunk_count, f"{path}.chunk_count")
        if self.chunk_count == 0 or self.chunk_count != len(self.chunk_slices):
            raise SchemaError("must equal the non-zero chunk_slices length", path=f"{path}.chunk_count")
        if tuple(chunk.chunk_id for chunk in self.chunk_slices) != tuple(range(self.chunk_count)):
            raise SchemaError("chunks must be canonically ordered by chunk_id", path=f"{path}.chunk_slices")
        if tuple(program.rank for program in self.rank_programs) != tuple(range(len(self.rank_programs))):
            raise SchemaError("rank programs must be canonically ordered by rank", path=f"{path}.rank_programs")
        rank_count = len(self.rank_programs)
        if self.chunk_dim is not ChunkDim.M:
            raise SchemaError(
                "naive fused GEMM+ReduceScatter requires M chunks",
                path=f"{path}.chunk_dim",
            )
        if self.chunk_count != rank_count or tuple(
            chunk.owner_rank for chunk in self.chunk_slices
        ) != tuple(range(rank_count)):
            raise SchemaError(
                "naive direct requires one canonical chunk per rank with owner_rank=chunk_id",
                path=f"{path}.chunk_slices",
            )
        if self.physical_output_layout != self.logical_output_layout:
            raise SchemaError(
                "naive direct physical output layout must equal logical output layout",
                path=f"{path}.physical_output_layout",
            )
        _validate_rank_programs(self.rank_programs, self.chunk_slices, path=path)
        if self.collective_algorithm is not CollectiveAlgorithm.DIRECT:
            raise SchemaError(
                "structural validator is implemented only for DIRECT naive",
                path=f"{path}.collective_algorithm",
            )
        _validate_direct_fusion_execution(self.rank_programs, self.chunk_slices, path=path)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunk_slices}
        expected_chunk_ids = set(chunk_by_id)
        forward_chunk_ids: set[int] = set()
        inverse_chunk_ids: set[int] = set()
        forward_triples: set[tuple[int, int, int]] = set()
        for index, entry in enumerate(self.output_permutation):
            entry.validate(f"{path}.output_permutation[{index}]")
            chunk = chunk_by_id.get(entry.chunk_id)
            if chunk is None:
                raise SchemaError("permutation references an unknown chunk", path=f"{path}.output_permutation[{index}].chunk_id")
            if entry.logical_owner_rank >= rank_count or entry.physical_owner_rank >= rank_count:
                raise SchemaError("permutation rank is out of range", path=f"{path}.output_permutation[{index}]")
            if entry.chunk_id in forward_chunk_ids:
                raise SchemaError("output permutation has duplicate chunk", path=f"{path}.output_permutation")
            if entry.physical_owner_rank != chunk.owner_rank:
                raise SchemaError("physical owner must equal chunk execution owner", path=f"{path}.output_permutation[{index}].physical_owner_rank")
            forward_chunk_ids.add(entry.chunk_id)
            forward_triples.add((entry.chunk_id, entry.physical_owner_rank, entry.logical_owner_rank))
        inverse_triples: set[tuple[int, int, int]] = set()
        for index, entry in enumerate(self.inverse_permutation):
            entry.validate(f"{path}.inverse_permutation[{index}]")
            chunk = chunk_by_id.get(entry.chunk_id)
            if chunk is None or entry.chunk_id in inverse_chunk_ids:
                raise SchemaError("inverse must name each known chunk once", path=f"{path}.inverse_permutation[{index}]")
            if entry.physical_owner_rank != chunk.owner_rank:
                raise SchemaError("inverse physical owner must equal chunk execution owner", path=f"{path}.inverse_permutation[{index}].physical_owner_rank")
            inverse_chunk_ids.add(entry.chunk_id)
            inverse_triples.add((entry.chunk_id, entry.physical_owner_rank, entry.logical_owner_rank))
        if forward_chunk_ids != expected_chunk_ids or inverse_chunk_ids != expected_chunk_ids:
            raise SchemaError("permutation and inverse must cover every chunk exactly once", path=f"{path}.output_permutation")
        if forward_triples != inverse_triples:
            raise SchemaError("inverse_permutation is not the inverse mapping", path=f"{path}.inverse_permutation")
        expected_forward = tuple(
            PermutationEntry(chunk_id, chunk_id, chunk_id)
            for chunk_id in range(rank_count)
        )
        expected_inverse = tuple(
            InversePermutationEntry(chunk_id, chunk_id, chunk_id)
            for chunk_id in range(rank_count)
        )
        if self.output_permutation != expected_forward or self.inverse_permutation != expected_inverse:
            raise SchemaError(
                "naive direct requires canonical identity permutation and inverse",
                path=f"{path}.output_permutation",
            )
        bindings: set[tuple[str, str]] = set()
        for index, binding in enumerate(self.consumer_layout_bindings):
            binding.validate(f"{path}.consumer_layout_bindings[{index}]")
            if binding.applies_inverse_permutation:
                raise SchemaError(
                    "naive identity layout forbids inverse permutation",
                    path=f"{path}.consumer_layout_bindings[{index}].applies_inverse_permutation",
                )
            key = (binding.consumer_node_id, binding.value_id)
            if key in bindings:
                raise SchemaError("duplicate consumer/value binding", path=f"{path}.consumer_layout_bindings[{index}]")
            bindings.add(key)
        expected_id = stable_artifact_id("fusion_plan", self._semantic_key(), schema_version=FUSION_PLAN_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(self, ir1: "IR1", path: str = "fusion_plan") -> None:
        self.validate(path)
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("plan references a different IR-1", path=f"{path}.source_ir1_id")
        groups = {group.id: group for group in ir1.groups}
        group = groups.get(self.group_ref)
        if group is None:
            raise SchemaError("dangling physical group", path=f"{path}.group_ref")
        fused_ops = {item.id: item for item in ir1.fused_op_skeletons}
        fused_op = fused_ops.get(self.fused_op_id)
        if fused_op is None:
            raise SchemaError("dangling fused op", path=f"{path}.fused_op_id")
        if group.instance_id != fused_op.instance_id:
            raise SchemaError("group and fused op belong to different instances", path=path)
        expected_profile = _owner_profile(
            ir1,
            fused_op.instance_id,
            path=f"{path}.profile_key",
        )
        if self.profile_key != expected_profile:
            message = (
                "profile key disagrees with IR-1"
                if not ir1.instance_profiles
                else "profile key disagrees with owner instance"
            )
            raise SchemaError(message, path=f"{path}.profile_key")
        if {program.rank for program in self.rank_programs} != {
            placement.rank for placement in group.placements
        }:
            raise SchemaError(
                "rank programs must exactly equal physical group ranks",
                path=f"{path}.rank_programs",
            )
        values = {value.id: value for value in ir1.values}
        for index, chunk in enumerate(self.chunk_slices):
            value = values.get(chunk.value_id)
            if value is None:
                raise SchemaError("chunk references a dangling IR-1 value", path=f"{path}.chunk_slices[{index}].value_id")
            if len(chunk.shape) != len(value.shape) or any(
                chunk.offset[axis] + chunk.shape[axis] > value.shape[axis]
                for axis in range(len(chunk.shape))
            ):
                raise SchemaError("chunk lies outside its IR-1 value", path=f"{path}.chunk_slices[{index}]")
            element_bytes = 2 if value.dtype is DType.FP16 else 4
            if chunk.bytes != math.prod(chunk.shape) * element_bytes:
                raise SchemaError("chunk byte count disagrees with shape/dtype", path=f"{path}.chunk_slices[{index}].bytes")
        if len(fused_op.boundary_outputs) != 1:
            raise SchemaError(
                "DIRECT naive currently requires one fused boundary output",
                path=f"{path}.fused_op_id",
            )
        boundary_output = values[fused_op.boundary_outputs[0]]
        _validate_chunk_cover(self.chunk_slices, boundary_output, self.chunk_dim, path=path)
        nodes = {node.id: node for node in ir1.nodes}
        member_nodes = [nodes[node_id] for node_id in fused_op.member_node_ids]
        if len(member_nodes) != 2:
            raise SchemaError(
                "DIRECT fused v1 requires exactly GEMM and ReduceScatter members",
                path=f"{path}.fused_op_id",
            )
        gemm_member, ordered_collective_member = member_nodes
        if (
            gemm_member.kind is not OpKind.GEMM
            or not isinstance(gemm_member.workload, GemmWorkload)
            or gemm_member.workload.partition is not GemmPartition.ROW_PARALLEL
            or ordered_collective_member.kind is not OpKind.COLLECTIVE
            or not isinstance(ordered_collective_member.workload, CollectiveWorkload)
            or ordered_collective_member.workload.collective
            is not CollectiveKind.REDUCE_SCATTER
        ):
            raise SchemaError(
                "fused member order must be GEMM then ReduceScatter",
                path=f"{path}.fused_op_id",
            )
        if (
            gemm_member.execution_group_ref != self.group_ref
            or ordered_collective_member.execution_group_ref != self.group_ref
        ):
            raise SchemaError(
                "both fused members must execute on the plan group",
                path=f"{path}.group_ref",
            )
        if (
            fused_op.boundary_inputs != gemm_member.inputs
            or fused_op.boundary_outputs != ordered_collective_member.outputs
        ):
            raise SchemaError(
                "fused boundaries must exactly equal GEMM inputs and ReduceScatter outputs",
                path=f"{path}.fused_op_id",
            )
        internal_data_edges = tuple(
            edge
            for edge in ir1.edges
            if edge.kind is EdgeKind.DATA
            and edge.source_node in fused_op.member_node_ids
            and edge.destination_node in fused_op.member_node_ids
        )
        if (
            len(internal_data_edges) != 1
            or internal_data_edges[0].source_node != gemm_member.id
            or internal_data_edges[0].destination_node
            != ordered_collective_member.id
            or gemm_member.outputs != ordered_collective_member.inputs
            or gemm_member.outputs != (internal_data_edges[0].value_id,)
        ):
            raise SchemaError(
                "fused members require one internal DATA edge GEMM to ReduceScatter",
                path=f"{path}.fused_op_id",
            )
        reduce_scatter_members = [
            node
            for node in member_nodes
            if node.kind is OpKind.COLLECTIVE
            and isinstance(node.workload, CollectiveWorkload)
            and node.workload.collective is CollectiveKind.REDUCE_SCATTER
        ]
        if len(reduce_scatter_members) != 1:
            raise SchemaError(
                "DIRECT fused v1 requires exactly one ReduceScatter member",
                path=f"{path}.fused_op_id",
            )
        collective_member = reduce_scatter_members[0]
        collective_workload = collective_member.workload
        collective_input = values[collective_member.inputs[0]]
        collective_output = values[collective_member.outputs[0]]
        if (
            collective_workload.dtype is not collective_input.dtype
            or collective_workload.dtype is not collective_output.dtype
            or collective_workload.input_layout != collective_input.logical_layout
            or collective_workload.output_layout != collective_output.logical_layout
        ):
            raise SchemaError(
                "ReduceScatter workload dtype/layout disagree with IR-1 values",
                path=f"{path}.fused_op_id",
            )
        if (
            sum(chunk.bytes for chunk in self.chunk_slices)
            != collective_workload.logical_tensor_bytes
        ):
            raise SchemaError("chunk bytes must equal collective workload bytes", path=f"{path}.chunk_slices")
        rank_count = len(self.rank_programs)
        if (
            collective_workload.participant_count != rank_count
            or collective_workload.scatter_tensor_axis != 0
            or boundary_output.shape[0] % rank_count
        ):
            raise SchemaError(
                "naive fused ReduceScatter requires an evenly divisible M scatter over all ranks",
                path=f"{path}.chunk_slices",
            )
        chunk_m = boundary_output.shape[0] // rank_count
        element_bytes = 2 if boundary_output.dtype is DType.FP16 else 4
        expected_chunk_bytes = math.prod((chunk_m, *boundary_output.shape[1:])) * element_bytes
        for chunk_id, chunk in enumerate(self.chunk_slices):
            expected_offset = (chunk_id * chunk_m,) + (0,) * (len(boundary_output.shape) - 1)
            expected_shape = (chunk_m,) + boundary_output.shape[1:]
            if (
                chunk.offset != expected_offset
                or chunk.shape != expected_shape
                or chunk.bytes != expected_chunk_bytes
                or chunk.bytes != collective_workload.rank_output_bytes
            ):
                raise SchemaError(
                    "naive fused chunks must be uniform canonical rank-output M slices",
                    path=f"{path}.chunk_slices[{chunk_id}]",
                )
        if (
            self.input_layout != collective_workload.input_layout
            or self.logical_output_layout != collective_workload.output_layout
        ):
            raise SchemaError(
                "plan layouts disagree with ReduceScatter member",
                path=f"{path}.input_layout",
            )
        comp_member_ids = {
            action.member_id
            for program in self.rank_programs
            for action in program.actions
            if action.kind is FusionActionKind.COMP
        }
        comp_member_id = next(iter(comp_member_ids)) if len(comp_member_ids) == 1 else None
        if (
            comp_member_id is None
            or comp_member_id not in nodes
            or nodes[comp_member_id].kind is not OpKind.GEMM
        ):
            raise SchemaError(
                "DIRECT fused v1 requires one GEMM contribution member",
                path=f"{path}.rank_programs",
            )
        rank_to_die = {placement.rank: placement.die_id for placement in group.placements}
        rank_placements = {placement.rank: placement for placement in group.placements}
        if group.axis is not MeshAxisName.TP or len(group.logical_shape) != 1:
            raise SchemaError(
                "DIRECT row-GEMM requires a one-dimensional TP group",
                path=f"{path}.group_ref",
            )
        state_manifest = ir1.persistent_state_manifest
        state_declarations = (
            {} if state_manifest is None else {item.id: item for item in state_manifest.declarations}
        )
        route_keys = {
            (route.source_rank, route.destination_rank, route.die_path)
            for route in group.embedding.routes
        }
        for program in self.rank_programs:
            if program.rank not in rank_to_die:
                raise SchemaError("rank program is absent from group placement", path=f"{path}.rank_programs")
            for action in program.actions:
                if action.member_id is not None and action.member_id not in fused_op.member_node_ids:
                    raise SchemaError("action references a non-member node", path=f"{path}.rank_programs")
                if (
                    action.kind is not FusionActionKind.COMP
                    and action.member_id != collective_member.id
                ):
                    raise SchemaError(
                        "transport/reduction action must reference the ReduceScatter member",
                        path=f"{path}.rank_programs",
                    )
                if action.compute is not None:
                    member = next(
                        node for node in ir1.nodes if node.id == action.member_id
                    )
                    if (
                        action.compute.op_kind,
                        action.compute.math,
                        action.compute.effects,
                        action.compute.impl_ref,
                    ) != (
                        member.kind,
                        member.math,
                        member.effects,
                        member.impl_ref,
                    ):
                        raise SchemaError(
                            "compute contract disagrees with fused member",
                            path=f"{path}.rank_programs",
                        )
                    if len(member.inputs) != 2 or len(member.outputs) != 1:
                        raise SchemaError(
                            "chunk-local GEMM requires explicit A, B, and partial output values",
                            path=f"{path}.fused_op_id",
                        )
                    full_m, full_n, full_k = member.workload.logical_shape
                    rank_m, rank_n, rank_k = member.workload.rank_shape
                    chunk = next(item for item in self.chunk_slices if item.id == action.slice_ref)
                    if (
                        (full_m, full_n) != boundary_output.shape
                        or (rank_m, rank_n) != boundary_output.shape
                        or values[member.inputs[0]].shape != (full_m, full_k)
                        or values[member.inputs[1]].shape != (full_k, full_n)
                        or values[member.outputs[0]].shape != (full_m, full_n)
                    ):
                        raise SchemaError(
                            "GEMM A/B/partial tensor shapes disagree with its workload",
                            path=f"{path}.fused_op_id",
                        )
                    expected_workload = GemmWorkload(
                        logical_shape=(chunk.shape[0], full_n, full_k),
                        rank_shape=(chunk.shape[0], rank_n, rank_k),
                        partition=member.workload.partition,
                        dtype=member.workload.dtype,
                    )
                    if len(action.compute.inputs) != 2 or len(action.compute.outputs) != 1:
                        raise SchemaError(
                            "chunk-local GEMM requires exactly two inputs and one output",
                            path=f"{path}.rank_programs",
                        )
                    rhs_operand_id = action.compute.inputs[1].value_id
                    if state_manifest is not None:
                        matching_accesses = tuple(
                            access
                            for access in ir1.state_accesses
                            if access.node_ref == member.id
                            and access.rank == program.rank
                            and state_declarations[access.state_ref].identity.kind
                            is StateKind.PARAMETER
                            and state_declarations[
                                access.state_ref
                            ].identity.tensor_ref
                            == member.inputs[1]
                        )
                        if len(matching_accesses) != 1:
                            raise SchemaError(
                                "stateful COMP requires exactly one rank-local RHS parameter access",
                                path=f"{path}.rank_programs",
                            )
                        rhs_operand_id = canonical_state_staging_value_id(
                            matching_accesses[0].id
                        )
                        if action.compute.inputs[1].value_id != rhs_operand_id:
                            raise SchemaError(
                                "COMP RHS operand must equal its canonical state staging value",
                                path=f"{path}.rank_programs",
                            )
                    placement = rank_placements[program.rank]
                    if len(placement.logical_coord) != 1:
                        raise SchemaError(
                            "TP rank requires one logical coordinate",
                            path=f"{path}.rank_programs",
                        )
                    rhs_k_offset = placement.logical_coord[0] * rank_k
                    expected_tile = ComputeTileBinding(
                        origin_workload=member.workload,
                        input_slices=(
                            ComputeOperandSlice(
                                action.compute.inputs[0].value_id,
                                member.inputs[0],
                                (chunk.offset[0], 0),
                                (chunk.shape[0], full_k),
                            ),
                            ComputeOperandSlice(
                                rhs_operand_id,
                                member.inputs[1],
                                (rhs_k_offset, 0),
                                (rank_k, full_n),
                            ),
                        ),
                        output_slices=(
                            ComputeOperandSlice(
                                action.compute.outputs[0].value_id,
                                member.outputs[0],
                                chunk.offset,
                                chunk.shape,
                            ),
                        ),
                    )
                    if action.compute.workload != expected_workload or action.compute.tile != expected_tile:
                        raise SchemaError(
                            "COMP must carry the exact chunk-local GEMM workload and A/B/partial logical slices",
                            path=f"{path}.rank_programs",
                        )
                if action.slice_ref is not None:
                    chunk = next(item for item in self.chunk_slices if item.id == action.slice_ref)
                    if action.dtype is not values[chunk.value_id].dtype:
                        raise SchemaError(
                            "slice action dtype must equal chunk value dtype",
                            path=f"{path}.rank_programs",
                        )
                if action.reduction is not None and (
                    action.reduction.reduce_op,
                    action.reduction.input_dtype,
                    action.reduction.accumulation_dtype,
                    action.reduction.output_dtype,
                ) != (
                    collective_workload.reduce_op,
                    collective_workload.dtype,
                    collective_member.math.accumulation_dtype,
                    boundary_output.dtype,
                ):
                    raise SchemaError(
                        "reduction contract disagrees with ReduceScatter semantics",
                        path=f"{path}.rank_programs",
                    )
                if action.peer_rank is not None:
                    route_ranks = (
                        (program.rank, action.peer_rank)
                        if action.kind is FusionActionKind.SEND
                        else (action.peer_rank, program.rank)
                    )
                    if (*route_ranks, action.expected_route) not in route_keys:
                        raise SchemaError("action route disagrees with group embedding", path=f"{path}.rank_programs")
                if action.kind in (FusionActionKind.COMP, FusionActionKind.RECV):
                    for temporary in action.writes:
                        if temporary in values:
                            raise SchemaError(
                                "contribution operand ids must be action-local temporaries, not IR-1 value ids",
                                path=f"{path}.rank_programs",
                            )
        expected_bindings = {
            (consumer, value_id)
            for value_id in fused_op.boundary_outputs
            for consumer in values[value_id].consumers
            if consumer not in fused_op.member_node_ids
        }
        actual_bindings = {
            (binding.consumer_node_id, binding.value_id)
            for binding in self.consumer_layout_bindings
        }
        if actual_bindings != expected_bindings:
            raise SchemaError(
                "consumer bindings must exactly cover external boundary-output consumers",
                path=f"{path}.consumer_layout_bindings",
            )
        for index, binding in enumerate(self.consumer_layout_bindings):
            expected_layout = (
                self.logical_output_layout
                if binding.applies_inverse_permutation
                else self.physical_output_layout
            )
            if binding.accepted_layout != expected_layout:
                raise SchemaError(
                    "consumer accepted_layout disagrees with permutation choice",
                    path=f"{path}.consumer_layout_bindings[{index}].accepted_layout",
                )


@dataclass(frozen=True, slots=True)
class StandaloneCollectivePlan:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    op_id: str
    algorithm: CollectiveAlgorithm
    group_ref: str
    profile_key: ProfileKey
    chunk_dim: ChunkDim
    chunk_slices: tuple[ChunkSlice, ...]
    rank_programs: tuple[RankProgram, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "StandaloneCollectivePlan":
        return cls(
            schema_version=STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("standalone_collective_plan", semantic_key, schema_version=STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "op_id", "algorithm", "group_ref", "profile_key",
            "chunk_dim", "chunk_slices", "rank_programs",
        )}

    def validate(self, path: str = "standalone_collective_plan") -> None:
        if self.schema_version != STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for field_name in ("producer_pass", "source_ir1_id", "op_id", "group_ref"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        self.profile_key.validate(f"{path}.profile_key")
        if not self.chunk_slices:
            raise SchemaError("must contain chunks", path=f"{path}.chunk_slices")
        if tuple(chunk.chunk_id for chunk in self.chunk_slices) != tuple(range(len(self.chunk_slices))):
            raise SchemaError("chunks must be canonically ordered by chunk_id", path=f"{path}.chunk_slices")
        if tuple(program.rank for program in self.rank_programs) != tuple(range(len(self.rank_programs))):
            raise SchemaError("rank programs must be canonically ordered by rank", path=f"{path}.rank_programs")
        rank_count = len(self.rank_programs)
        if len(self.chunk_slices) != rank_count or tuple(
            chunk.owner_rank for chunk in self.chunk_slices
        ) != tuple(range(rank_count)):
            raise SchemaError(
                "naive direct AllGather requires one canonical chunk per rank",
                path=f"{path}.chunk_slices",
            )
        _validate_rank_programs(self.rank_programs, self.chunk_slices, path=path)
        if self.algorithm is not CollectiveAlgorithm.DIRECT:
            raise SchemaError(
                "structural validator is implemented only for DIRECT AllGather",
                path=f"{path}.algorithm",
            )
        _validate_direct_all_gather_execution(self.rank_programs, self.chunk_slices, path=path)
        expected_id = stable_artifact_id("standalone_collective_plan", self._semantic_key(), schema_version=STANDALONE_COLLECTIVE_PLAN_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(self, ir1: "IR1", path: str = "standalone_collective_plan") -> None:
        self.validate(path)
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("plan references a different IR-1", path=f"{path}.source_ir1_id")
        groups = {group.id: group for group in ir1.groups}
        group = groups.get(self.group_ref)
        nodes = {node.id: node for node in ir1.nodes}
        op = nodes.get(self.op_id)
        if group is None or op is None:
            raise SchemaError("dangling group or collective op", path=path)
        if group.instance_id != op.instance_id or group.mesh_ref != op.mesh_ref:
            raise SchemaError("group and op placement disagree", path=path)
        expected_profile = _owner_profile(
            ir1,
            op.instance_id,
            path=f"{path}.profile_key",
        )
        if self.profile_key != expected_profile:
            message = (
                "profile key disagrees with IR-1"
                if not ir1.instance_profiles
                else "profile key disagrees with owner instance"
            )
            raise SchemaError(message, path=f"{path}.profile_key")
        if self.group_ref != op.execution_group_ref:
            raise SchemaError(
                "standalone group must equal the op execution group",
                path=f"{path}.group_ref",
            )
        if op.kind is not OpKind.COLLECTIVE or not isinstance(op.workload, CollectiveWorkload):
            raise SchemaError("standalone plan requires a collective node", path=f"{path}.op_id")
        if op.workload.collective is not CollectiveKind.ALL_GATHER:
            raise SchemaError(
                "standalone DIRECT v1 supports only AllGather",
                path=f"{path}.op_id",
            )
        if {program.rank for program in self.rank_programs} != {
            placement.rank for placement in group.placements
        }:
            raise SchemaError(
                "rank programs must exactly equal physical group ranks",
                path=f"{path}.rank_programs",
            )
        values = {value.id: value for value in ir1.values}
        for index, chunk in enumerate(self.chunk_slices):
            value = values.get(chunk.value_id)
            if value is None:
                raise SchemaError("chunk references a dangling IR-1 value", path=f"{path}.chunk_slices[{index}].value_id")
            if len(chunk.shape) != len(value.shape) or any(
                chunk.offset[axis] + chunk.shape[axis] > value.shape[axis]
                for axis in range(len(chunk.shape))
            ):
                raise SchemaError("chunk lies outside its IR-1 value", path=f"{path}.chunk_slices[{index}]")
            element_bytes = 2 if value.dtype is DType.FP16 else 4
            if chunk.bytes != math.prod(chunk.shape) * element_bytes:
                raise SchemaError("chunk byte count disagrees with shape/dtype", path=f"{path}.chunk_slices[{index}].bytes")
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise SchemaError(
                "standalone DIRECT AllGather requires one input and one output",
                path=f"{path}.op_id",
            )
        input_value = values[op.inputs[0]]
        output_value = values[op.outputs[0]]
        if (input_value.shape, input_value.dtype) != (output_value.shape, output_value.dtype):
            raise SchemaError("AllGather input/output shape and dtype must match", path=f"{path}.op_id")
        if (
            op.workload.dtype is not input_value.dtype
            or op.workload.dtype is not output_value.dtype
            or op.workload.input_layout != input_value.logical_layout
            or op.workload.output_layout != output_value.logical_layout
        ):
            raise SchemaError(
                "AllGather workload dtype/layout disagree with IR-1 values",
                path=f"{path}.op_id",
            )
        _validate_chunk_cover(self.chunk_slices, output_value, self.chunk_dim, path=path)
        axis = 0 if self.chunk_dim is ChunkDim.M else 1
        if (
            op.workload.gather_tensor_axis != axis
            or output_value.shape[axis] % len(self.rank_programs)
            or input_value.sharding.dim_map[axis] is not group.axis
            or output_value.sharding.dim_map[axis] is not None
            or input_value.sharding.partial
            or output_value.sharding.partial
        ):
            raise SchemaError(
                "AllGather chunks and input/output sharding must match the gather axis",
                path=f"{path}.chunk_dim",
            )
        chunk_extent = output_value.shape[axis] // len(self.rank_programs)
        for chunk_id, chunk in enumerate(self.chunk_slices):
            expected_offset = tuple(
                chunk_id * chunk_extent if index == axis else 0
                for index in range(len(output_value.shape))
            )
            expected_shape = tuple(
                chunk_extent if index == axis else extent
                for index, extent in enumerate(output_value.shape)
            )
            if (
                chunk.offset != expected_offset
                or chunk.shape != expected_shape
                or chunk.bytes != op.workload.rank_input_bytes
            ):
                raise SchemaError(
                    "AllGather requires uniform canonical owner input shards",
                    path=f"{path}.chunk_slices[{chunk_id}]",
                )
        if (
            sum(chunk.bytes for chunk in self.chunk_slices)
            != op.workload.logical_tensor_bytes
        ):
            raise SchemaError("chunk bytes must equal collective workload bytes", path=f"{path}.chunk_slices")
        route_keys = {
            (route.source_rank, route.destination_rank, route.die_path)
            for route in group.embedding.routes
        }
        ranks = {placement.rank for placement in group.placements}
        for program in self.rank_programs:
            if program.rank not in ranks:
                raise SchemaError("rank program is absent from group placement", path=f"{path}.rank_programs")
            for action in program.actions:
                if action.member_id != op.id:
                    raise SchemaError(
                        "standalone action member must equal collective op",
                        path=f"{path}.rank_programs",
                    )
                if action.slice_ref is not None:
                    chunk = next(item for item in self.chunk_slices if item.id == action.slice_ref)
                    if action.dtype is not output_value.dtype:
                        raise SchemaError("slice action dtype must equal chunk value dtype", path=f"{path}.rank_programs")
                if action.kind is FusionActionKind.LOCAL_COPY and (
                    action.reads != op.inputs or action.writes != op.outputs
                ):
                    raise SchemaError("LOCAL_COPY must read input and write output", path=f"{path}.rank_programs")
                if action.kind is FusionActionKind.SEND and action.reads != op.outputs:
                    raise SchemaError("AllGather SEND must read output value", path=f"{path}.rank_programs")
                if action.kind is FusionActionKind.RECV and action.writes != op.outputs:
                    raise SchemaError("AllGather RECV must write output value", path=f"{path}.rank_programs")
                if action.peer_rank is not None:
                    route_ranks = (
                        (program.rank, action.peer_rank)
                        if action.kind is FusionActionKind.SEND
                        else (action.peer_rank, program.rank)
                    )
                    if (*route_ranks, action.expected_route) not in route_keys:
                        raise SchemaError("action route disagrees with group embedding", path=f"{path}.rank_programs")
