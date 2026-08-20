"""Per-core relocatable command fragments (linking follows in N1d3-b2)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from enum import Enum, IntEnum

from ..errors import SchemaError
from .action import (
    BarrierScope,
    ComputeContract,
    FusionPlan,
    StandaloneCollectivePlan,
)
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import (
    STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY,
    GlobalAction,
    GlobalActionDAG,
    LogicalCoreRef,
)
from .ir0 import (
    AttentionMode,
    AttentionWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyBackwardWorkload,
    CrossEntropyReduction,
    EmbeddingTablePlacement,
    EmbeddingWorkload,
    ElementwiseWorkload,
    GemmWorkload,
    GreedySampleWorkload,
    OpKind,
    PackedQkvLayout,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    SampleRowSelection,
    SamplingMode,
    SgdUpdateWorkload,
    SwiGluWorkload,
)
from .ir1 import IR1
from .ir2 import (
    BufferOwnership,
    BufferUseRole,
    FusedNodeOrigin,
    FlowRouteRole,
    IR2ProjectionResult,
    IntraDieScheduleSet,
    RegionLowering,
    SemanticTaskKind,
    StandaloneNodeOrigin,
    StateTransferOrigin,
    TensorSlice,
    dense_row_major_view_byte_addend,
)
from .persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)


STATE_ABI_SCHEMA_VERSION = "wafer_frontend.state_abi/v1alpha1"
COMMAND_FRAGMENT_SCHEMA_VERSION = "wafer_frontend.command_fragment/v1alpha13"
REGION_MANIFEST_SCHEMA_VERSION = "wafer_frontend.region_manifest/v1alpha12"
LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION = "wafer_frontend.linked_program_manifest/v1alpha14"
PLAN_BARRIER_RUNTIME_SYMBOL_SCHEMA_VERSION = (
    "wafer_frontend.plan_barrier_runtime_symbol/v1"
)
STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION = (
    "wafer_frontend.state_transfer_wave_runtime_symbol/v1"
)
_DTE_ENDPOINT_P2P_MAX_BYTES = ((1 << 16) - 1) * 16
_COMPUTE_PARAMETER_MAX = (1 << 30) - 1


class FragmentKind(str, Enum):
    COARSE = "coarse"
    STANDALONE_COLLECTIVE = "standalone_collective"
    ISA_REGION = "isa_region"
    STATE_IO = "state_io"
    STATE_TRANSFER = "state_transfer"
    MOE_TRANSFER = "moe_transfer"
    S2_LITE_ROOTED_AR = "s2_lite_rooted_ar"


_FRAGMENT_KIND_BY_LOWERING = {
    RegionLowering.JSON_COARSE: FragmentKind.COARSE,
    RegionLowering.STRICT_ACTIONS: FragmentKind.STANDALONE_COLLECTIVE,
    RegionLowering.ISA_REGION: FragmentKind.ISA_REGION,
    RegionLowering.STRICT_STATE_IO: FragmentKind.STATE_IO,
    RegionLowering.STRICT_STATE_TRANSFER: FragmentKind.STATE_TRANSFER,
}


class RecordOpcode(IntEnum):
    """Stable public Opcode values consumed by the C++ program artifact."""

    MATMUL = 0x01
    ATTENTION = 0x06
    SWIGLU = 0x0C
    RESIDUAL = 0x0E
    RMSNORM = 0x10
    ROPE_QK_EXACT = 0x1A
    ATTENTION_EXACT = 0x1B
    EMBEDDING_LOOKUP = 0x1C
    GREEDY_SAMPLE = 0x1D
    CROSS_ENTROPY_FORWARD = 0x1E
    CROSS_ENTROPY_BACKWARD = 0x1F
    SGD_UPDATE = 0x20
    DTE_SEND = 0x40
    DTE_RECV = 0x41
    LOCAL_REDUCE = 0x43
    DTE_ISSUE = 0x82
    SRAM_BIND = 0x84
    LSU_LOAD = 0x80
    LSU_STORE = 0x81
    SRAM_FREE = 0x86
    SRAM_ALLOC_AT = 0x89
    DTE_WAIT = 0xC0
    EVENT_SET = 0xC3
    EVENT_WAIT = 0xC4


class RuntimeSymbolKind(str, Enum):
    START_TAG = "start_tag"
    EVENT_TAG = "event_tag"
    DTE_TOKEN = "dte_token"
    DTE_FSM = "dte_fsm"
    GROUP = "group"
    RUNTIME_CORE = "runtime_core"


class PlanBarrierEventPhase(str, Enum):
    ARRIVE = "arrive"
    RELEASE = "release"


class ProgramSymbolKind(IntEnum):
    ABSOLUTE_ADDRESS = 1
    SRAM_REGION = 2
    SRAM_LABEL = 3


class RuntimeOperandField(str, Enum):
    START_TAG = "start_tag"
    EVENT_TAG = "event_tag"
    DTE_TOKEN = "dte_token"
    DTE_FSM = "dte_fsm"
    GROUP_ID = "group_id"
    SOURCE_CORE = "source_core"
    DESTINATION_CORE = "destination_core"
    PEER_CORE = "peer_core"


_RUNTIME_KIND_BY_FIELD = {
    RuntimeOperandField.START_TAG: RuntimeSymbolKind.START_TAG,
    RuntimeOperandField.EVENT_TAG: RuntimeSymbolKind.EVENT_TAG,
    RuntimeOperandField.DTE_TOKEN: RuntimeSymbolKind.DTE_TOKEN,
    RuntimeOperandField.DTE_FSM: RuntimeSymbolKind.DTE_FSM,
    RuntimeOperandField.GROUP_ID: RuntimeSymbolKind.GROUP,
    RuntimeOperandField.SOURCE_CORE: RuntimeSymbolKind.RUNTIME_CORE,
    RuntimeOperandField.DESTINATION_CORE: RuntimeSymbolKind.RUNTIME_CORE,
    RuntimeOperandField.PEER_CORE: RuntimeSymbolKind.RUNTIME_CORE,
}
_RUNTIME_FIELD_ORDER = {field: index for index, field in enumerate(RuntimeOperandField)}


class SemanticOperandId(IntEnum):
    COMPUTE_INPUT_ADDRESS = 1
    COMPUTE_DATA_ADDRESS = 2
    COMPUTE_OUTPUT_ADDRESS = 3
    SOURCE_ADDRESS = 4
    DESTINATION_ADDRESS = 5
    HBM_ADDRESS = 6
    SYMBOL = 7
    REGION_NAME = 8
    LABEL_SYMBOL = 9
    OLD_SYMBOL = 10
    NEW_SYMBOL = 11
    COMPUTE_AUX_ADDRESS = 12
    SRAM_BIND_INPUT_0 = 0x100
    SRAM_BIND_INPUT_1 = 0x101
    SRAM_BIND_INPUT_2 = 0x102
    SRAM_BIND_INPUT_3 = 0x103
    SRAM_BIND_INPUT_4 = 0x104
    SRAM_BIND_INPUT_5 = 0x105
    SRAM_BIND_INPUT_6 = 0x106
    SRAM_BIND_INPUT_7 = 0x107
    SRAM_BIND_INPUT_8 = 0x108
    SRAM_BIND_INPUT_9 = 0x109
    SRAM_BIND_INPUT_10 = 0x10A
    SRAM_BIND_INPUT_11 = 0x10B
    SRAM_BIND_INPUT_12 = 0x10C
    SRAM_BIND_INPUT_13 = 0x10D
    SRAM_BIND_INPUT_14 = 0x10E
    SRAM_BIND_INPUT_15 = 0x10F
    SRAM_BIND_OUTPUT = 0x110


class OperandKind(str, Enum):
    LITERAL = "literal"
    RUNTIME_SYMBOL = "runtime_symbol"
    ADDRESS_SYMBOL = "address_symbol"


LiteralValue = int | str | bool | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSymbol:
    """Logical symbol; source_ref is the schedule binding ref for runtime kinds.

    DTE_FSM/RUNTIME_CORE use ``channel_symbol``; DTE_TOKEN uses
    ``token_symbol``; EVENT_TAG/SOURCE_CORE/DESTINATION_CORE use the producing
    ``event_symbol``. START_TAG and GROUP use their top-level provenance key.
    """

    id: str
    kind: RuntimeSymbolKind
    source_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.source_ref, f"{path}.source_ref")


def _plan_barrier_action_fields(
    action: GlobalAction, path: str
) -> tuple[str, int, object, LogicalCoreRef]:
    if (
        action.task_kind is not SemanticTaskKind.BARRIER
        or not isinstance(action.origin_ref, StandaloneNodeOrigin)
        or action.sync is None
        or action.sync.barrier is None
        or action.sync.barrier.scope is not BarrierScope.PLAN
        or action.logical_core is None
        or action.runtime_binding is None
        or action.runtime_binding.event_symbol != action.sync.barrier.id
    ):
        raise SchemaError(
            "PLAN barrier runtime symbols require one placed standalone PLAN barrier action",
            path=path,
        )
    return (
        action.origin_ref.collective_plan_id,
        action.origin_ref.rank,
        action.sync.barrier,
        action.logical_core,
    )


def canonical_plan_barrier_core_symbol(
    source_global_dag_id: str, action: GlobalAction
) -> RuntimeSymbol:
    """Return the barrier-local runtime-core symbol for one participant action."""

    validate_nonempty(source_global_dag_id, "source_global_dag_id")
    plan_id, rank, barrier, logical_core = _plan_barrier_action_fields(
        action, "action"
    )
    return RuntimeSymbol(
        stable_artifact_id(
            "plan_barrier_core",
            {
                "source_global_dag_id": source_global_dag_id,
                "collective_plan_id": plan_id,
                "barrier_id": barrier.id,
                "rank": rank,
                "action_id": action.id,
                "logical_core": logical_core,
            },
            schema_version=PLAN_BARRIER_RUNTIME_SYMBOL_SCHEMA_VERSION,
        ),
        RuntimeSymbolKind.RUNTIME_CORE,
        barrier.id,
    )


def canonical_plan_barrier_event_symbol(
    source_global_dag_id: str,
    phase: PlanBarrierEventPhase,
    source: GlobalAction,
    destination: GlobalAction,
) -> RuntimeSymbol:
    """Return one directed coordinator-arrival or -release event symbol."""

    validate_nonempty(source_global_dag_id, "source_global_dag_id")
    if not isinstance(phase, PlanBarrierEventPhase):
        raise SchemaError("must be a PlanBarrierEventPhase", path="phase")
    source_plan, source_rank, source_barrier, source_core = (
        _plan_barrier_action_fields(source, "source")
    )
    destination_plan, destination_rank, destination_barrier, destination_core = (
        _plan_barrier_action_fields(destination, "destination")
    )
    if (
        source_plan != destination_plan
        or source_barrier != destination_barrier
        or source_rank == destination_rank
        or source_core.die_id == destination_core.die_id
    ):
        raise SchemaError(
            "PLAN barrier event endpoints must be distinct ranks on distinct dies in one exact plan barrier",
            path="source",
        )
    return RuntimeSymbol(
        stable_artifact_id(
            "plan_barrier_event",
            {
                "source_global_dag_id": source_global_dag_id,
                "collective_plan_id": source_plan,
                "barrier_id": source_barrier.id,
                "phase": phase,
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "source_action_id": source.id,
                "destination_action_id": destination.id,
                "source_core": source_core,
                "destination_core": destination_core,
            },
            schema_version=PLAN_BARRIER_RUNTIME_SYMBOL_SCHEMA_VERSION,
        ),
        RuntimeSymbolKind.EVENT_TAG,
        source_barrier.id,
    )


def canonical_state_transfer_wave_symbols(
    source_global_dag_id: str,
    source: GlobalAction,
    destination: GlobalAction,
) -> tuple[RuntimeSymbol, RuntimeSymbol, RuntimeSymbol]:
    """Return exact source-core, destination-core and tag wave symbols."""

    validate_nonempty(source_global_dag_id, "source_global_dag_id")
    source_origin = source.origin_ref
    destination_origin = destination.origin_ref
    if (
        source.task_kind is not SemanticTaskKind.WAIT
        or destination.task_kind is not SemanticTaskKind.SEND
        or not isinstance(source_origin, StateTransferOrigin)
        or not isinstance(destination_origin, StateTransferOrigin)
        or source_origin.segment_index is None
        or destination_origin.segment_index is None
        or source.logical_core is None
        or destination.logical_core is None
        or source.logical_core == destination.logical_core
        or source.id not in destination.deps
    ):
        raise SchemaError(
            "state-transfer wave must be one cross-core segmented WAIT-to-SEND dependency",
            path="source",
        )
    binding_ref = stable_artifact_id(
        "state_transfer_wave_binding",
        {
            "source_global_dag_id": source_global_dag_id,
            "source_action_id": source.id,
            "destination_action_id": destination.id,
            "capacity": STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY,
        },
        schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
    )
    def core_symbol(role: str, action: GlobalAction) -> RuntimeSymbol:
        return RuntimeSymbol(
            stable_artifact_id(
                "state_transfer_wave_core",
                {
                    "binding_ref": binding_ref,
                    "role": role,
                    "logical_core": action.logical_core,
                },
                schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
            ),
            RuntimeSymbolKind.RUNTIME_CORE,
            binding_ref,
        )

    return (
        core_symbol("source", source),
        core_symbol("destination", destination),
        RuntimeSymbol(
            stable_artifact_id(
                "state_transfer_wave_event",
                {
                    "binding_ref": binding_ref,
                    "source_action_id": source.id,
                    "destination_action_id": destination.id,
                },
                schema_version=STATE_TRANSFER_WAVE_RUNTIME_SYMBOL_SCHEMA_VERSION,
            ),
            RuntimeSymbolKind.EVENT_TAG,
            binding_ref,
        ),
    )
def state_transfer_wave_action_maps(
    actions: dict[str, GlobalAction],
    path: str,
) -> tuple[dict[str, GlobalAction], dict[str, GlobalAction]]:
    """Index the unique segmented WAIT-to-SEND cross-core wave edges."""

    incoming_by_send: dict[str, GlobalAction] = {}
    outgoing_by_wait: dict[str, GlobalAction] = {}
    for destination in actions.values():
        destination_origin = destination.origin_ref
        if (
            destination.task_kind is not SemanticTaskKind.SEND
            or not isinstance(destination_origin, StateTransferOrigin)
            or destination_origin.segment_index is None
        ):
            continue
        sources = tuple(
            actions[dependency]
            for dependency in destination.deps
            if dependency in actions
            and actions[dependency].task_kind is SemanticTaskKind.WAIT
            and isinstance(
                actions[dependency].origin_ref,
                StateTransferOrigin,
            )
            and actions[dependency].origin_ref.segment_index is not None
            and actions[dependency].logical_core != destination.logical_core
        )
        if len(sources) > 1:
            raise SchemaError(
                "segmented SEND has multiple wave predecessors",
                path=path,
            )
        if sources:
            source = sources[0]
            incoming_by_send[destination.id] = source
            if source.id in outgoing_by_wait:
                raise SchemaError(
                    "segmented WAIT has multiple wave successors",
                    path=path,
                )
            outgoing_by_wait[source.id] = destination
    return incoming_by_send, outgoing_by_wait


def validate_state_transfer_wave_fragment_shape(
    actions: tuple[GlobalAction, ...],
    incoming_by_send: dict[str, GlobalAction],
    outgoing_by_wait: dict[str, GlobalAction],
    path: str,
) -> None:
    """Require one exact capacity-three segmented endpoint fragment shape."""

    if not actions or any(
        not isinstance(action.origin_ref, StateTransferOrigin)
        or action.origin_ref.segment_index is None
        for action in actions
    ):
        return
    capacity = STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY
    if all(action.task_kind is SemanticTaskKind.SEND for action in actions):
        flags = tuple(action.id in incoming_by_send for action in actions)
        count = len(flags)
        plain = min(capacity, count)
        suffix = (False,) * plain + (True,) * (count - plain)
        all_wait = (True,) * count
        transition = (
            (True,)
            + (False,) * min(capacity - 1, count - 1)
            + (True,) * max(0, count - capacity)
        )
        if flags not in (suffix, all_wait, transition):
            raise SchemaError(
                "segmented source wave records must be plain capacity prefix, all-wave suffix continuation, or one transition wait followed by a capacity window",
                path=path,
            )
        return

    waits = tuple(
        action for action in actions if action.task_kind is SemanticTaskKind.WAIT
    )
    if len(waits) * 2 != len(actions):
        return
    flags = tuple(action.id in outgoing_by_wait for action in waits)
    count = len(flags)
    all_set = (True,) * count
    prefix = (
        (True,) * max(0, count - capacity)
        + (False,) * min(capacity, count)
    )
    transition_tail = (
        (True,) * (count - capacity)
        + (False,) * (capacity - 1)
        + (True,)
        if count >= capacity
        else ()
    )
    if flags not in (all_set, prefix, transition_tail):
        raise SchemaError(
            "segmented destination wave records must be one canonical capacity-three prefix or terminal transition shape",
            path=path,
        )


def canonical_state_transfer_wave_record(
    source_global_dag_id: str,
    *,
    owner: GlobalAction,
    source: GlobalAction,
    destination: GlobalAction,
    opcode: RecordOpcode,
) -> RelocatableRecord:
    source_core, destination_core, event = (
        canonical_state_transfer_wave_symbols(
            source_global_dag_id, source, destination
        )
    )
    if (
        opcode is RecordOpcode.EVENT_SET
        and owner.id != source.id
        or opcode is RecordOpcode.EVENT_WAIT
        and owner.id != destination.id
    ):
        raise SchemaError(
            "wave EVENT owner does not match dependency direction",
            path="owner",
        )
    operands = (
        RecordOperand.runtime(
            "source_core",
            RuntimeOperandField.SOURCE_CORE,
            source_core.id,
        ),
        RecordOperand.runtime(
            "destination_core",
            RuntimeOperandField.DESTINATION_CORE,
            destination_core.id,
        ),
        RecordOperand.runtime(
            "tag",
            RuntimeOperandField.EVENT_TAG,
            event.id,
        ),
    )
    if opcode is RecordOpcode.EVENT_WAIT:
        operands = (*operands, RecordOperand.literal("count", 1))
    return RelocatableRecord(owner.id, opcode, operands)




@dataclass(frozen=True, slots=True)
class ProgramSymbol:
    id: str
    kind: ProgramSymbolKind
    source_ref: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.source_ref, f"{path}.source_ref")


def _validate_literal(value: LiteralValue, path: str) -> None:
    if type(value) is int:
        validate_uint64(value, path)
    elif type(value) is str:
        validate_nonempty(value, path)
    elif type(value) is bool:
        return
    elif type(value) is tuple:
        for index, item in enumerate(value):
            validate_uint64(item, f"{path}[{index}]")
    else:
        raise SchemaError("unsupported literal type", path=path)


@dataclass(frozen=True, slots=True)
class RecordOperand:
    name: str
    kind: OperandKind
    literal_value: LiteralValue | None
    runtime_field: RuntimeOperandField | None
    operand_id: SemanticOperandId | None
    symbol_ref: str | None

    @classmethod
    def literal(cls, name: str, value: LiteralValue) -> "RecordOperand":
        return cls(name, OperandKind.LITERAL, value, None, None, None)

    @classmethod
    def runtime(
        cls, name: str, field: RuntimeOperandField, symbol_ref: str
    ) -> "RecordOperand":
        return cls(name, OperandKind.RUNTIME_SYMBOL, None, field, None, symbol_ref)

    @classmethod
    def address(
        cls, name: str, operand_id: SemanticOperandId, symbol_ref: str
    ) -> "RecordOperand":
        return cls(name, OperandKind.ADDRESS_SYMBOL, None, None, operand_id, symbol_ref)

    def validate(self, path: str) -> None:
        validate_nonempty(self.name, f"{path}.name")
        if self.kind is OperandKind.LITERAL:
            if self.literal_value is None or any(
                value is not None for value in (self.runtime_field, self.operand_id, self.symbol_ref)
            ):
                raise SchemaError("literal operand carries non-literal state", path=path)
            _validate_literal(self.literal_value, f"{path}.literal_value")
        elif self.kind is OperandKind.RUNTIME_SYMBOL:
            if self.runtime_field is None or self.symbol_ref is None or self.literal_value is not None or self.operand_id is not None:
                raise SchemaError("runtime-symbol operand is incomplete or mixed", path=path)
            validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")
        else:
            if self.operand_id is None or self.symbol_ref is None or self.literal_value is not None or self.runtime_field is not None:
                raise SchemaError("address-symbol operand is incomplete or mixed", path=path)
            validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")


@dataclass(frozen=True, slots=True)
class _OperandSpec:
    name: str
    allowed_kinds: tuple[OperandKind, ...]
    runtime_field: RuntimeOperandField | None = None
    operand_id: SemanticOperandId | None = None


def _lit(name: str) -> _OperandSpec:
    return _OperandSpec(name, (OperandKind.LITERAL,))


def _run(name: str, field: RuntimeOperandField, *, literal_allowed: bool = False) -> _OperandSpec:
    kinds = (OperandKind.LITERAL, OperandKind.RUNTIME_SYMBOL) if literal_allowed else (OperandKind.RUNTIME_SYMBOL,)
    return _OperandSpec(name, kinds, runtime_field=field)


def _addr(name: str, operand_id: SemanticOperandId) -> _OperandSpec:
    return _OperandSpec(name, (OperandKind.ADDRESS_SYMBOL,), operand_id=operand_id)


def _optional_label(name: str, operand_id: SemanticOperandId) -> _OperandSpec:
    return _OperandSpec(
        name,
        (OperandKind.LITERAL, OperandKind.ADDRESS_SYMBOL),
        operand_id=operand_id,
    )


_SRAM_BIND_INPUT_OPERAND_IDS = tuple(
    SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index)
    for index in range(16)
)
_SRAM_BIND_OPERANDS = (
    _lit("input_count"),
    *(
        _optional_label(f"input_label_{index}", operand_id)
        for index, operand_id in enumerate(_SRAM_BIND_INPUT_OPERAND_IDS)
    ),
    _addr("output_label", SemanticOperandId.SRAM_BIND_OUTPUT),
)


_COMPUTE_OPERANDS_WITH_DATA = (
    _lit("datatype"),
    _addr("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("parameters"),
)

_COMPUTE_OPERANDS_UNUSED_DATA = (
    _lit("datatype"),
    _addr("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _lit("data_address"),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("parameters"),
)


_ROPE_QK_EXACT_OPERANDS = (
    _lit("datatype"),
    _lit("packed_layout"),
    _addr("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("logical_tokens"),
    _lit("tp_degree"),
    _lit("num_heads"),
    _lit("num_kv_heads"),
    _lit("rank_num_heads"),
    _lit("rank_num_kv_heads"),
    _lit("head_dim"),
    _lit("rotary_dim"),
    _lit("max_position_embeddings"),
    _lit("context_max"),
    _lit("rope_theta_f64_bits"),
)

_ATTENTION_EXACT_OPERANDS = (
    _lit("datatype"),
    _lit("mode"),
    _lit("packed_layout"),
    _lit("causal"),
    _addr("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("query_tokens"),
    _lit("tp_degree"),
    _lit("num_heads"),
    _lit("num_kv_heads"),
    _lit("rank_num_heads"),
    _lit("rank_num_kv_heads"),
    _lit("head_dim"),
    _lit("context_sum"),
    _lit("context_max"),
    _lit("query_key_pairs"),
    _lit("rank_kv_read_bytes"),
    _lit("rank_kv_write_bytes"),
)

_EMBEDDING_LOOKUP_OPERANDS = (
    _lit("index_datatype"),
    _lit("table_datatype"),
    _lit("output_datatype"),
    _lit("placement"),
    _addr("indices_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("table_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("logical_rows"),
    _lit("rank_rows"),
    _lit("tp_degree"),
    _lit("vocab_size"),
    _lit("hidden_size"),
)

_GREEDY_SAMPLE_OPERANDS = (
    _lit("logits_datatype"),
    _lit("output_datatype"),
    _lit("mode"),
    _lit("row_selection"),
    _addr("logits_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("tp_degree"),
    _lit("token_rows"),
    _lit("vocab_size"),
    _lit("sample_count"),
    _lit("comparisons"),
)

_CROSS_ENTROPY_FORWARD_OPERANDS = (
    _lit("logits_datatype"),
    _lit("label_datatype"),
    _lit("loss_datatype"),
    _lit("reduction"),
    _addr("logits_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("labels_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
    _addr("loss_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("logical_rows"),
    _lit("rank_rows"),
    _lit("tp_degree"),
    _lit("vocab_size"),
)

_CROSS_ENTROPY_BACKWARD_OPERANDS = (
    _lit("logits_datatype"),
    _lit("label_datatype"),
    _lit("upstream_datatype"),
    _lit("output_datatype"),
    _lit("reduction"),
    _lit("upstream_mode"),
    _addr("logits_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("labels_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
    _addr("upstream_address", SemanticOperandId.COMPUTE_AUX_ADDRESS),
    _addr("logits_grad_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("logical_rows"),
    _lit("rank_rows"),
    _lit("tp_degree"),
    _lit("vocab_size"),
    _lit("upstream_elements"),
)

_SGD_UPDATE_OPERANDS = (
    _lit("weight_datatype"),
    _lit("gradient_datatype"),
    _lit("output_datatype"),
    _lit("rounding"),
    _addr("weight_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
    _addr("gradient_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
    _addr("updated_weight_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
    _lit("element_count"),
    _lit("learning_rate_f64_bits"),
    _lit("momentum_f64_bits"),
)


_OPERAND_SCHEMAS = {
    RecordOpcode.MATMUL: _COMPUTE_OPERANDS_WITH_DATA,
    RecordOpcode.ATTENTION: _COMPUTE_OPERANDS_UNUSED_DATA,
    RecordOpcode.SWIGLU: _COMPUTE_OPERANDS_UNUSED_DATA,
    RecordOpcode.RESIDUAL: _COMPUTE_OPERANDS_WITH_DATA,
    RecordOpcode.RMSNORM: _COMPUTE_OPERANDS_WITH_DATA,
    RecordOpcode.ROPE_QK_EXACT: _ROPE_QK_EXACT_OPERANDS,
    RecordOpcode.ATTENTION_EXACT: _ATTENTION_EXACT_OPERANDS,
    RecordOpcode.EMBEDDING_LOOKUP: _EMBEDDING_LOOKUP_OPERANDS,
    RecordOpcode.GREEDY_SAMPLE: _GREEDY_SAMPLE_OPERANDS,
    RecordOpcode.CROSS_ENTROPY_FORWARD: _CROSS_ENTROPY_FORWARD_OPERANDS,
    RecordOpcode.CROSS_ENTROPY_BACKWARD: _CROSS_ENTROPY_BACKWARD_OPERANDS,
    RecordOpcode.SGD_UPDATE: _SGD_UPDATE_OPERANDS,
    RecordOpcode.SRAM_BIND: _SRAM_BIND_OPERANDS,
    RecordOpcode.SRAM_FREE: (
        _addr("symbol", SemanticOperandId.SYMBOL),
    ),
    RecordOpcode.SRAM_ALLOC_AT: (
        _addr("region_name", SemanticOperandId.REGION_NAME),
        _addr("label_symbol", SemanticOperandId.LABEL_SYMBOL),
        _lit("region_offset_bytes"),
        _lit("size_bytes"),
        _lit("alignment_bytes"),
        _lit("lifetime"),
        _lit("spillable"),
    ),
    RecordOpcode.DTE_SEND: (
        _lit("mode"), _lit("source_space"), _lit("completion"), _lit("datatype"),
        _lit("reduce_op"), _run("fsm_id", RuntimeOperandField.DTE_FSM),
        _run("token", RuntimeOperandField.DTE_TOKEN, literal_allowed=True), _lit("length_bytes"),
        _addr("source_address", SemanticOperandId.SOURCE_ADDRESS),
        _run("peer_core", RuntimeOperandField.PEER_CORE), _lit("expected_sources"),
        _lit("tree_id"), _run("group_id", RuntimeOperandField.GROUP_ID, literal_allowed=True),
        _lit("collective_id"), _lit("epoch"),
    ),
    RecordOpcode.DTE_RECV: (
        _lit("mode"), _lit("completion"), _lit("datatype"), _lit("reduce_op"),
        _run("fsm_id", RuntimeOperandField.DTE_FSM),
        _run("token", RuntimeOperandField.DTE_TOKEN, literal_allowed=True), _lit("length_bytes"),
        _addr("destination_address", SemanticOperandId.DESTINATION_ADDRESS),
        _run("peer_core", RuntimeOperandField.PEER_CORE), _lit("expected_sources"),
        _lit("tree_id"), _run("group_id", RuntimeOperandField.GROUP_ID, literal_allowed=True),
        _lit("collective_id"), _lit("epoch"),
    ),
    RecordOpcode.LOCAL_REDUCE: (
        _lit("input_dtype"), _lit("accumulator_dtype"), _lit("output_dtype"),
        _lit("reduce_op"), _lit("rounding"), _lit("order"), _lit("input_count"),
        _lit("element_count"), _lit("input_stride_bytes"),
        _addr("source_address", SemanticOperandId.SOURCE_ADDRESS),
        _addr("destination_address", SemanticOperandId.DESTINATION_ADDRESS),
    ),
    RecordOpcode.LSU_LOAD: (
        _addr("hbm_address", SemanticOperandId.HBM_ADDRESS),
        _lit("size_bytes"),
        _addr("destination_address", SemanticOperandId.DESTINATION_ADDRESS),
    ),
    RecordOpcode.LSU_STORE: (
        _addr("hbm_address", SemanticOperandId.HBM_ADDRESS),
        _lit("size_bytes"),
        _addr("source_address", SemanticOperandId.SOURCE_ADDRESS),
    ),

    RecordOpcode.DTE_ISSUE: (
        _lit("direction"), _run("token", RuntimeOperandField.DTE_TOKEN),
        _lit("payload_bits"), _lit("size_bytes"), _lit("hbm_address"),
        _addr("source_address", SemanticOperandId.SOURCE_ADDRESS),
        _addr("destination_address", SemanticOperandId.DESTINATION_ADDRESS),
    ),
    RecordOpcode.DTE_WAIT: (_run("token", RuntimeOperandField.DTE_TOKEN),),
    RecordOpcode.EVENT_SET: (
        _run("source_core", RuntimeOperandField.SOURCE_CORE),
        _run("destination_core", RuntimeOperandField.DESTINATION_CORE),
        _run("tag", RuntimeOperandField.EVENT_TAG),
    ),
    RecordOpcode.EVENT_WAIT: (
        _run("source_core", RuntimeOperandField.SOURCE_CORE),
        _run("destination_core", RuntimeOperandField.DESTINATION_CORE),
        _run("tag", RuntimeOperandField.EVENT_TAG), _lit("count"),
    ),
}


_ALLOWED_ADDRESS_KINDS = {
    **{
        (RecordOpcode.SRAM_BIND, operand_id): (ProgramSymbolKind.SRAM_LABEL,)
        for operand_id in (
            *_SRAM_BIND_INPUT_OPERAND_IDS,
            SemanticOperandId.SRAM_BIND_OUTPUT,
        )
    },
    (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS, ProgramSymbolKind.SRAM_REGION),
    (RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS, ProgramSymbolKind.SRAM_REGION),
    (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.LSU_LOAD, SemanticOperandId.HBM_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.LSU_LOAD, SemanticOperandId.DESTINATION_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.LSU_STORE, SemanticOperandId.HBM_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.LSU_STORE, SemanticOperandId.SOURCE_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS,),
    (RecordOpcode.DTE_ISSUE, SemanticOperandId.SOURCE_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS, ProgramSymbolKind.SRAM_REGION),
    (RecordOpcode.DTE_ISSUE, SemanticOperandId.DESTINATION_ADDRESS): (ProgramSymbolKind.ABSOLUTE_ADDRESS, ProgramSymbolKind.SRAM_REGION),
    (RecordOpcode.SRAM_FREE, SemanticOperandId.SYMBOL): (
        ProgramSymbolKind.SRAM_LABEL,
    ),
    (RecordOpcode.SRAM_ALLOC_AT, SemanticOperandId.REGION_NAME): (
        ProgramSymbolKind.SRAM_REGION,
    ),
    (RecordOpcode.SRAM_ALLOC_AT, SemanticOperandId.LABEL_SYMBOL): (
        ProgramSymbolKind.SRAM_LABEL,
    ),
}

for _compute_opcode in (
    RecordOpcode.ATTENTION,
    RecordOpcode.SWIGLU,
    RecordOpcode.RMSNORM,
    RecordOpcode.ROPE_QK_EXACT,
    RecordOpcode.ATTENTION_EXACT,
    RecordOpcode.GREEDY_SAMPLE,
):
    for _operand_id in (
        SemanticOperandId.COMPUTE_INPUT_ADDRESS,
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ):
        _ALLOWED_ADDRESS_KINDS[(_compute_opcode, _operand_id)] = (
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
        )

for _compute_opcode in (
    RecordOpcode.RESIDUAL,
    RecordOpcode.RMSNORM,
    RecordOpcode.EMBEDDING_LOOKUP,
    RecordOpcode.CROSS_ENTROPY_FORWARD,
    RecordOpcode.SGD_UPDATE,
):
    for _operand_id in (
        SemanticOperandId.COMPUTE_INPUT_ADDRESS,
        SemanticOperandId.COMPUTE_DATA_ADDRESS,
        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    ):
        _ALLOWED_ADDRESS_KINDS[(_compute_opcode, _operand_id)] = (
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
        )

for _operand_id in (
    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
    SemanticOperandId.COMPUTE_DATA_ADDRESS,
    SemanticOperandId.COMPUTE_AUX_ADDRESS,
    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
):
    _ALLOWED_ADDRESS_KINDS[
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, _operand_id)
    ] = (ProgramSymbolKind.ABSOLUTE_ADDRESS,)


_COMPUTE_OPCODE_BY_IMPL_REF = {
    "matmul_forward": (OpKind.GEMM, RecordOpcode.MATMUL),
    "lm_head_wgrad": (OpKind.GEMM, RecordOpcode.MATMUL),
    "attention": (OpKind.ATTENTION, RecordOpcode.ATTENTION),
    "attention_forward": (OpKind.ATTENTION, RecordOpcode.ATTENTION_EXACT),
    "rope_qk_exact": (OpKind.ROPE, RecordOpcode.ROPE_QK_EXACT),
    "embedding_lookup": (OpKind.EMBEDDING, RecordOpcode.EMBEDDING_LOOKUP),
    "greedy_sample": (OpKind.SAMPLING, RecordOpcode.GREEDY_SAMPLE),
    "cross_entropy_forward": (
        OpKind.CE_FORWARD,
        RecordOpcode.CROSS_ENTROPY_FORWARD,
    ),
    "cross_entropy_backward": (
        OpKind.CE_BACKWARD,
        RecordOpcode.CROSS_ENTROPY_BACKWARD,
    ),
    "sgd_update": (
        OpKind.OPTIMIZER_UPDATE,
        RecordOpcode.SGD_UPDATE,
    ),
    "swiglu": (OpKind.ELEMENTWISE, RecordOpcode.SWIGLU),
    "residual": (OpKind.ELEMENTWISE, RecordOpcode.RESIDUAL),
    "rms_norm": (OpKind.NORM, RecordOpcode.RMSNORM),
    "rmsnorm": (OpKind.NORM, RecordOpcode.RMSNORM),
}

_COMPUTE_PARAMETER_COUNTS = {
    RecordOpcode.MATMUL: 4,
    RecordOpcode.ATTENTION: 5,
    RecordOpcode.SWIGLU: 1,
    RecordOpcode.RESIDUAL: 1,
    RecordOpcode.RMSNORM: 3,
}


@dataclass(frozen=True, slots=True)
class _ComputeRecordABI:
    opcode: RecordOpcode
    bind_input_count: int
    parameters: tuple[int, ...]
    data_input_index: int | None
    aux_input_index: int | None = None


def _shape_elements(shape: tuple[int, ...], *, path: str) -> int:
    elements = 1
    for extent in shape:
        elements *= extent
    if elements == 0 or elements > _COMPUTE_PARAMETER_MAX:
        raise SchemaError(
            "rank-local flattened extent must be positive and fit the 30-bit compute ABI",
            path=path,
        )
    return elements

_FIXED_COMPUTE_OPCODES = (
    RecordOpcode.ROPE_QK_EXACT,
    RecordOpcode.ATTENTION_EXACT,
    RecordOpcode.EMBEDDING_LOOKUP,
    RecordOpcode.GREEDY_SAMPLE,
    RecordOpcode.CROSS_ENTROPY_FORWARD,
    RecordOpcode.CROSS_ENTROPY_BACKWARD,
    RecordOpcode.SGD_UPDATE,
)


def _fixed_compute_literals(
    compute: ComputeContract,
    opcode: RecordOpcode,
    *,
    path: str,
) -> dict[str, int]:
    workload = compute.workload
    if opcode is RecordOpcode.ROPE_QK_EXACT:
        if (
            type(workload) is not RopeQkWorkload
            or len(compute.inputs) != 1
            or workload.dtype is not DType.FP16
            or workload.packed_layout is not PackedQkvLayout.Q_K_V
        ):
            raise SchemaError(
                "ROPE_QK_EXACT requires one packed-QKV FP16 input/output",
                path=path,
            )
        return {
            "datatype": 1,
            "packed_layout": 0,
            "logical_tokens": workload.profile.prefill_tokens
            + workload.profile.decode_tokens,
            "tp_degree": workload.num_heads // workload.rank_num_heads,
            "num_heads": workload.num_heads,
            "num_kv_heads": workload.num_kv_heads,
            "rank_num_heads": workload.rank_num_heads,
            "rank_num_kv_heads": workload.rank_num_kv_heads,
            "head_dim": workload.head_dim,
            "rotary_dim": workload.rotary_dim,
            "max_position_embeddings": workload.max_position_embeddings,
            "context_max": workload.profile.context_max,
            "rope_theta_f64_bits": struct.unpack(
                "<Q", struct.pack("<d", workload.rope_theta)
            )[0],
        }
    if opcode is RecordOpcode.ATTENTION_EXACT:
        if (
            type(workload) is not AttentionWorkload
            or len(compute.inputs) != 1
            or workload.dtype is not DType.FP16
        ):
            raise SchemaError(
                "ATTENTION_EXACT requires one typed FP16 attention input/output",
                path=path,
            )
        return {
            "datatype": 1,
            "mode": (
                2
                if workload.exact_profile is not None
                else (
                    0
                    if workload.mode is AttentionMode.PREFILL
                    else (
                        1
                        if workload.mode is AttentionMode.DECODE
                        else 3
                        if workload.mode is AttentionMode.TRAIN_FORWARD
                        else -1
                    )
                )
            ),
            "packed_layout": 0,
            "causal": True,
            "query_tokens": workload.query_tokens,
            "tp_degree": workload.num_heads // workload.rank_num_heads,
            "num_heads": workload.num_heads,
            "num_kv_heads": workload.num_kv_heads,
            "rank_num_heads": workload.rank_num_heads,
            "rank_num_kv_heads": workload.rank_num_kv_heads,
            "head_dim": workload.head_dim,
            "context_sum": workload.context_sum,
            "context_max": workload.context_max,
            "query_key_pairs": workload.query_key_pairs,
            "rank_kv_read_bytes": workload.rank_kv_read_bytes,
            "rank_kv_write_bytes": workload.rank_kv_write_bytes,
        }
    if opcode is RecordOpcode.EMBEDDING_LOOKUP:
        if (
            type(workload) is not EmbeddingWorkload
            or len(compute.inputs) != 2
            or workload.index_dtype is not DType.INT32
            or workload.table_dtype is not DType.FP16
            or workload.output_dtype is not DType.FP16
            or workload.table_placement is not EmbeddingTablePlacement.REPLICATED
        ):
            raise SchemaError(
                "EMBEDDING_LOOKUP requires INT32 indices and one replicated FP16 table",
                path=path,
            )
        return {
            "index_datatype": 2,
            "table_datatype": 1,
            "output_datatype": 1,
            "placement": 0,
            "logical_rows": workload.logical_index_shape[0],
            "rank_rows": workload.rank_index_shape[0],
            "tp_degree": workload.logical_index_shape[0]
            // workload.rank_index_shape[0],
            "vocab_size": workload.logical_table_shape[0],
            "hidden_size": workload.logical_table_shape[1],
        }
    if opcode is RecordOpcode.GREEDY_SAMPLE:
        if (
            type(workload) is not GreedySampleWorkload
            or len(compute.inputs) != 1
            or workload.logits_dtype is not DType.FP16
            or workload.output_dtype is not DType.INT32
            or workload.mode is not SamplingMode.GREEDY
            or workload.row_selection is not SampleRowSelection.LAST_PER_SEQUENCE
        ):
            raise SchemaError(
                "GREEDY_SAMPLE requires TP1 FP16 logits to INT32 sample ids",
                path=path,
            )
        return {
            "logits_datatype": 1,
            "output_datatype": 2,
            "mode": 0,
            "row_selection": 0,
            "tp_degree": workload.tp_degree,
            "token_rows": workload.logical_logits_shape[0],
            "vocab_size": workload.logical_logits_shape[1],
            "sample_count": workload.sample_count,
            "comparisons": workload.comparisons,
        }
    if opcode is RecordOpcode.CROSS_ENTROPY_FORWARD:
        if (
            type(workload) is not CrossEntropyForwardWorkload
            or len(compute.inputs) != 2
            or workload.logits_dtype is not DType.FP16
            or workload.label_dtype is not DType.INT32
            or workload.loss_dtype is not DType.FP32
            or workload.reduction is not CrossEntropyReduction.NONE
        ):
            raise SchemaError(
                "CROSS_ENTROPY_FORWARD requires unreduced FP16 logits, INT32 labels, and FP32 loss",
                path=path,
            )
        return {
            "logits_datatype": 1,
            "label_datatype": 2,
            "loss_datatype": 3,
            "reduction": 0,
            "logical_rows": workload.logical_logits_shape[0],
            "rank_rows": workload.rank_logits_shape[0],
            "tp_degree": workload.logical_logits_shape[0]
            // workload.rank_logits_shape[0],
            "vocab_size": workload.logical_logits_shape[1],
        }
    if opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD:
        if (
            type(workload) is not CrossEntropyBackwardWorkload
            or len(compute.inputs) != 3
            or workload.logits_dtype is not DType.FP16
            or workload.label_dtype is not DType.INT32
            or workload.loss_gradient_dtype is not DType.FP32
            or workload.logits_gradient_dtype is not DType.FP16
            or workload.reduction is not CrossEntropyReduction.NONE
        ):
            raise SchemaError(
                "CROSS_ENTROPY_BACKWARD requires unreduced FP16 logits, INT32 labels, FP32 per-row upstream, and FP16 logits gradient",
                path=path,
            )
        return {
            "logits_datatype": 1,
            "label_datatype": 2,
            "upstream_datatype": 3,
            "output_datatype": 1,
            "reduction": 0,
            "upstream_mode": 1,
            "logical_rows": workload.logical_logits_shape[0],
            "rank_rows": workload.rank_logits_shape[0],
            "tp_degree": workload.logical_logits_shape[0]
            // workload.rank_logits_shape[0],
            "vocab_size": workload.logical_logits_shape[1],
            "upstream_elements": workload.rank_loss_gradient_shape[0],
        }
    if opcode is RecordOpcode.SGD_UPDATE:
        if (
            type(workload) is not SgdUpdateWorkload
            or len(compute.inputs) != 2
            or workload.weight_dtype is not DType.FP16
            or workload.gradient_dtype is not DType.FP32
            or workload.updated_weight_dtype is not DType.FP16
            or workload.momentum != 0.0
        ):
            raise SchemaError(
                "SGD_UPDATE requires FP16 weight/output, FP32 gradient, and momentum zero",
                path=path,
            )
        return {
            "weight_datatype": 1,
            "gradient_datatype": 3,
            "output_datatype": 1,
            "rounding": 0,
            "element_count": workload.element_count,
            "learning_rate_f64_bits": struct.unpack(
                "<Q", struct.pack("<d", workload.learning_rate)
            )[0],
            "momentum_f64_bits": 0,
        }
    raise SchemaError("opcode is not a fixed compute record", path=path)




def _validate_fixed_compute_operands(
    opcode: RecordOpcode,
    operands: tuple[RecordOperand, ...],
    *,
    path: str,
) -> None:
    values = {
        operand.name: operand.literal_value
        for operand in operands
        if operand.kind is OperandKind.LITERAL
    }

    def positive(*names: str) -> None:
        for name in names:
            value = values[name]
            if type(value) is not int or value <= 0:
                raise SchemaError(
                    "fixed compute geometry must be a positive integer",
                    path=f"{path}.{name}",
                )

    if opcode is RecordOpcode.ROPE_QK_EXACT:
        if values["datatype"] != 1 or values["packed_layout"] != 0:
            raise SchemaError(
                "ROPE_QK_EXACT requires FP16 and packed Q_K_V",
                path=path,
            )
        positive(
            "logical_tokens",
            "tp_degree",
            "num_heads",
            "num_kv_heads",
            "rank_num_heads",
            "rank_num_kv_heads",
            "head_dim",
            "rotary_dim",
            "max_position_embeddings",
            "context_max",
        )
        if (
            values["num_heads"] != values["tp_degree"] * values["rank_num_heads"]
            or values["num_kv_heads"]
            != values["tp_degree"] * values["rank_num_kv_heads"]
            or values["rotary_dim"] != values["head_dim"]
            or values["rotary_dim"] % 2
            or values["context_max"] > values["max_position_embeddings"]
        ):
            raise SchemaError("ROPE_QK_EXACT geometry is inconsistent", path=path)
        theta = struct.unpack(
            "<d", struct.pack("<Q", values["rope_theta_f64_bits"])
        )[0]
        if not math.isfinite(theta) or theta <= 0.0:
            raise SchemaError(
                "ROPE_QK_EXACT theta bits must encode a finite positive f64",
                path=f"{path}.rope_theta_f64_bits",
            )
        return

    if opcode is RecordOpcode.ATTENTION_EXACT:
        if (
            values["datatype"] != 1
            or values["mode"] not in (0, 1, 2, 3)
            or values["packed_layout"] != 0
            or type(values["causal"]) is not bool
            or not values["causal"]
        ):
            raise SchemaError(
                "ATTENTION_EXACT requires FP16 packed causal attention",
                path=path,
            )
        positive(
            "query_tokens",
            "tp_degree",
            "num_heads",
            "num_kv_heads",
            "rank_num_heads",
            "rank_num_kv_heads",
            "head_dim",
            "context_sum",
            "context_max",
            "query_key_pairs",
        )
        if (
            values["num_heads"] != values["tp_degree"] * values["rank_num_heads"]
            or values["num_kv_heads"]
            != values["tp_degree"] * values["rank_num_kv_heads"]
        ):
            raise SchemaError("ATTENTION_EXACT head shards are inconsistent", path=path)
        if values["mode"] == 0:
            expected_pairs = (
                values["query_tokens"] * (values["query_tokens"] + 1) // 2
            )
            expected_read = 0
        elif values["mode"] == 1:
            expected_pairs = values["context_sum"]
            expected_read = (
                4
                * values["context_sum"]
                * values["rank_num_kv_heads"]
                * values["head_dim"]
            )
        elif values["mode"] == 2:
            bytes_per_token = (
                4 * values["rank_num_kv_heads"] * values["head_dim"]
            )
            if (
                values["context_sum"] < values["query_tokens"]
                or values["query_key_pairs"] < values["query_tokens"]
                or values["query_key_pairs"]
                > values["query_tokens"] * values["context_max"]
                or values["rank_kv_read_bytes"] % bytes_per_token
                or values["rank_kv_read_bytes"]
                > values["context_sum"] * bytes_per_token
            ):
                raise SchemaError(
                    "ATTENTION_EXACT static-profile metrics are inconsistent",
                    path=path,
                )
            expected_pairs = values["query_key_pairs"]
            expected_read = values["rank_kv_read_bytes"]
        else:
            expected_pairs = (
                values["query_tokens"] * (values["query_tokens"] + 1) // 2
            )
            expected_read = 0
        expected_write = (
            0
            if values["mode"] == 3
            else 4
            * values["query_tokens"]
            * values["rank_num_kv_heads"]
            * values["head_dim"]
        )
        if (
            values["query_key_pairs"] != expected_pairs
            or values["rank_kv_read_bytes"] != expected_read
            or values["rank_kv_write_bytes"] != expected_write
        ):
            raise SchemaError("ATTENTION_EXACT derived counts are inconsistent", path=path)
        return

    if opcode is RecordOpcode.EMBEDDING_LOOKUP:
        if (
            values["index_datatype"] != 2
            or values["table_datatype"] != 1
            or values["output_datatype"] != 1
            or values["placement"] != 0
        ):
            raise SchemaError(
                "EMBEDDING_LOOKUP requires INT32 indices and replicated FP16 data",
                path=path,
            )
        positive(
            "logical_rows",
            "rank_rows",
            "tp_degree",
            "vocab_size",
            "hidden_size",
        )
        if values["logical_rows"] != values["rank_rows"] * values["tp_degree"]:
            raise SchemaError("EMBEDDING_LOOKUP rows do not preserve TP", path=path)
        return

    if opcode is RecordOpcode.GREEDY_SAMPLE:
        if (
            values["logits_datatype"] != 1
            or values["output_datatype"] != 2
            or values["mode"] != 0
            or values["row_selection"] != 0
            or values["tp_degree"] != 1
        ):
            raise SchemaError(
                "GREEDY_SAMPLE requires TP1 FP16 logits and INT32 output",
                path=path,
            )
        positive("token_rows", "vocab_size", "sample_count", "comparisons")
        if (
            values["vocab_size"] <= 1
            or values["sample_count"] > values["token_rows"]
            or values["comparisons"]
            != values["sample_count"] * (values["vocab_size"] - 1)
        ):
            raise SchemaError("GREEDY_SAMPLE derived counts are inconsistent", path=path)
        return

    if opcode is RecordOpcode.CROSS_ENTROPY_FORWARD:
        if (
            values["logits_datatype"] != 1
            or values["label_datatype"] != 2
            or values["loss_datatype"] != 3
            or values["reduction"] != 0
        ):
            raise SchemaError(
                "CROSS_ENTROPY_FORWARD requires FP16/INT32/FP32 and reduction NONE",
                path=path,
            )
        positive("logical_rows", "rank_rows", "tp_degree", "vocab_size")
        if (
            values["vocab_size"] <= 1
            or values["logical_rows"]
            != values["rank_rows"] * values["tp_degree"]
        ):
            raise SchemaError(
                "CROSS_ENTROPY_FORWARD row partition or vocabulary is inconsistent",
                path=path,
            )
        return

    if opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD:
        if (
            values["logits_datatype"] != 1
            or values["label_datatype"] != 2
            or values["upstream_datatype"] != 3
            or values["output_datatype"] != 1
            or values["reduction"] != 0
            or values["upstream_mode"] not in (0, 1)
        ):
            raise SchemaError(
                "CROSS_ENTROPY_BACKWARD requires FP16/INT32/FP32/FP16, reduction NONE, and a known upstream mode",
                path=path,
            )
        positive(
            "logical_rows",
            "rank_rows",
            "tp_degree",
            "vocab_size",
            "upstream_elements",
        )
        expected_upstream = 1 if values["upstream_mode"] == 0 else values["rank_rows"]
        if (
            values["vocab_size"] <= 1
            or values["logical_rows"]
            != values["rank_rows"] * values["tp_degree"]
            or values["upstream_elements"] != expected_upstream
        ):
            raise SchemaError(
                "CROSS_ENTROPY_BACKWARD row partition, vocabulary, or upstream extent is inconsistent",
                path=path,
            )
        return

    if opcode is RecordOpcode.SGD_UPDATE:
        if (
            values["weight_datatype"] != 1
            or values["gradient_datatype"] != 3
            or values["output_datatype"] != 1
            or values["rounding"] != 0
            or values["momentum_f64_bits"] != 0
        ):
            raise SchemaError(
                "SGD_UPDATE requires FP16 weight/output, FP32 gradient, RNE, and zero momentum",
                path=path,
            )
        positive("element_count")
        learning_rate_bits = values["learning_rate_f64_bits"]
        if (
            type(learning_rate_bits) is not int
            or learning_rate_bits < 0
            or learning_rate_bits > (1 << 64) - 1
        ):
            raise SchemaError(
                "SGD_UPDATE learning-rate bits must be a uint64",
                path=f"{path}.learning_rate_f64_bits",
            )
        learning_rate = struct.unpack(
            "<d", struct.pack("<Q", learning_rate_bits)
        )[0]
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise SchemaError(
                "SGD_UPDATE learning-rate bits must encode a finite positive f64",
                path=f"{path}.learning_rate_f64_bits",
            )
        if operands[4].symbol_ref != operands[6].symbol_ref:
            raise SchemaError(
                "SGD_UPDATE weight and updated_weight must be the same in-place symbol",
                path=f"{path}.updated_weight_address",
            )
        return

    raise SchemaError("opcode is not a fixed compute record", path=path)

def _compute_record_abi(
    compute: ComputeContract,
    *,
    path: str,
) -> _ComputeRecordABI:
    mapping = _COMPUTE_OPCODE_BY_IMPL_REF.get(compute.impl_ref)
    if mapping is None:
        raise SchemaError(
            "compute impl_ref has no frozen public opcode mapping",
            path=f"{path}.impl_ref",
        )
    expected_kind, opcode = mapping
    if compute.op_kind is not expected_kind:
        raise SchemaError(
            "compute op_kind disagrees with its frozen impl_ref mapping",
            path=f"{path}.op_kind",
        )
    if len(compute.outputs) != 1:
        raise SchemaError(
            "Dense compute ABI requires exactly one output",
            path=f"{path}.outputs",
        )

    workload = compute.workload
    if opcode in _FIXED_COMPUTE_OPCODES:
        _fixed_compute_literals(compute, opcode, path=path)
        has_data_input = opcode in (
            RecordOpcode.EMBEDDING_LOOKUP,
            RecordOpcode.CROSS_ENTROPY_FORWARD,
            RecordOpcode.CROSS_ENTROPY_BACKWARD,
            RecordOpcode.SGD_UPDATE,
        )
        has_aux_input = opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD
        bind_input_count = 3 if has_aux_input else 2 if has_data_input else 1
        result = _ComputeRecordABI(
            opcode,
            bind_input_count,
            (),
            1 if has_data_input else None,
            2 if has_aux_input else None,
        )
    elif opcode is RecordOpcode.MATMUL:
        if (
            type(workload) is not GemmWorkload
            or workload.dtype is not DType.FP16
            or len(compute.inputs) != 2
        ):
            raise SchemaError(
                "MATMUL requires one FP16 rank-local two-input/one-output GEMM",
                path=path,
            )
        rank_m, rank_n, rank_k = workload.rank_shape
        parameters = (1, rank_m, rank_k, rank_n)
        result = _ComputeRecordABI(opcode, 1, parameters, 1)
    elif opcode is RecordOpcode.ATTENTION:
        if (
            type(workload) is not AttentionWorkload
            or workload.dtype is not DType.FP16
            or len(compute.inputs) != 1
            or bool(workload.profile.prefill_tokens)
            == bool(workload.profile.decode_tokens)
        ):
            raise SchemaError(
                "ATTENTION requires one FP16 input/output and exactly one prefill/decode token mode",
                path=path,
            )
        tokens = (
            workload.profile.prefill_tokens
            if workload.profile.prefill_tokens
            else workload.profile.decode_tokens
        )
        qkv_width = workload.head_dim * (
            workload.rank_num_heads + 2 * workload.rank_num_kv_heads
        )
        ratio = workload.rank_num_heads // workload.rank_num_kv_heads
        result = _ComputeRecordABI(
            opcode,
            1,
            (1, tokens, qkv_width, workload.rank_num_heads, ratio),
            None,
        )
    elif opcode is RecordOpcode.SWIGLU:
        if (
            type(workload) is not SwiGluWorkload
            or workload.dtype is not DType.FP16
            or len(compute.inputs) != 1
        ):
            raise SchemaError(
                "SWIGLU requires one FP16 concat input and one output",
                path=path,
            )
        logical_input = workload.logical_input_shape
        rank_input = workload.rank_input_shape
        if (
            logical_input[:-1] != workload.logical_output_shape[:-1]
            or logical_input[-1] != 2 * workload.logical_output_shape[-1]
            or rank_input[:-1] != workload.rank_output_shape[:-1]
            or rank_input[-1] != 2 * workload.rank_output_shape[-1]
        ):
            raise SchemaError(
                "SWIGLU input must be one concat tensor with final extent 2N",
                path=f"{path}.workload",
            )
        result = _ComputeRecordABI(
            opcode,
            1,
            (
                _shape_elements(
                    workload.rank_output_shape,
                    path=f"{path}.workload.rank_output_shape",
                ),
            ),
            None,
        )
    elif opcode is RecordOpcode.RESIDUAL:
        if (
            type(workload) is not ResidualWorkload
            or workload.dtype is not DType.FP16
            or len(compute.inputs) != 2
        ):
            raise SchemaError(
                "RESIDUAL requires two equal-shape FP16 inputs and one output",
                path=path,
            )
        result = _ComputeRecordABI(
            opcode,
            2,
            (
                _shape_elements(
                    workload.rank_shape,
                    path=f"{path}.workload.rank_output_shape",
                ),
            ),
            1,
        )
    else:
        if (
            opcode is not RecordOpcode.RMSNORM
            or type(workload) is not RmsNormWorkload
            or len(compute.inputs) != 2
        ):
            raise SchemaError(
                "RMSNORM requires one FP16 activation and one scale input",
                path=path,
            )
        rank_t, rank_c = workload.rank_activation_shape
        result = _ComputeRecordABI(opcode, 1, (1, rank_t, rank_c), 1)

    if any(
        type(value) is not int
        or value <= 0
        or value > _COMPUTE_PARAMETER_MAX
        for value in result.parameters
    ):
        raise SchemaError(
            "derived compute parameters must be positive and fit the 30-bit backend ABI",
            path=f"{path}.workload",
        )
    return result


@dataclass(frozen=True, slots=True)
class RelocatableRecord:
    source_global_action_id: str
    opcode: RecordOpcode
    operands: tuple[RecordOperand, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.source_global_action_id, f"{path}.source_global_action_id")
        expected = _OPERAND_SCHEMAS[self.opcode]
        if tuple(operand.name for operand in self.operands) != tuple(spec.name for spec in expected):
            raise SchemaError("operands must exactly follow the opcode's canonical ABI order", path=f"{path}.operands")
        for index, (operand, spec) in enumerate(zip(self.operands, expected)):
            operand.validate(f"{path}.operands[{index}]")
            if operand.kind not in spec.allowed_kinds:
                raise SchemaError("operand kind is illegal for opcode/name", path=f"{path}.operands[{index}].kind")
            if operand.kind is OperandKind.RUNTIME_SYMBOL and operand.runtime_field is not spec.runtime_field:
                raise SchemaError("runtime field is illegal for opcode/name", path=f"{path}.operands[{index}].runtime_field")
            if operand.kind is OperandKind.ADDRESS_SYMBOL and operand.operand_id is not spec.operand_id:
                raise SchemaError("semantic operand id is illegal for opcode/name", path=f"{path}.operands[{index}].operand_id")
        if self.opcode in _COMPUTE_PARAMETER_COUNTS:
            datatype = self.operands[0].literal_value
            parameters = self.operands[-1].literal_value
            if (
                datatype != 1
                or
                type(parameters) is not tuple
                or len(parameters) != _COMPUTE_PARAMETER_COUNTS[self.opcode]
                or any(value == 0 for value in parameters)
                or any(value > _COMPUTE_PARAMETER_MAX for value in parameters)
            ):
                raise SchemaError(
                    "compute requires FP16 datatype=1 and positive <=30-bit parameters with exact canonical count",
                    path=f"{path}.operands[{len(self.operands) - 1}].literal_value",
                )
            if (
                self.opcode
                in (
                    RecordOpcode.ATTENTION,
                    RecordOpcode.SWIGLU,
                )
                and self.operands[2].literal_value != 0
            ):
                raise SchemaError(
                    "manual-memory compute requires literal data_address=0 with no relocation",
                    path=f"{path}.operands[2].literal_value",
                )
        if self.opcode in _FIXED_COMPUTE_OPCODES:
            _validate_fixed_compute_operands(self.opcode, self.operands, path=path)
        if self.opcode in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
            size_bytes = self.operands[1].literal_value
            if type(size_bytes) is not int or size_bytes == 0:
                raise SchemaError(
                    "LSU size_bytes must be a non-zero uint64",
                    path=f"{path}.operands[1].literal_value",
                )

        if self.opcode is RecordOpcode.SRAM_BIND:
            input_count = self.operands[0].literal_value
            if type(input_count) is not int or not 1 <= input_count <= 16:
                raise SchemaError(
                    "SRAM_BIND input_count must be in the fixed [1,16] ABI range",
                    path=f"{path}.operands[0].literal_value",
                )
            for index, operand in enumerate(self.operands[1:17]):
                if index < input_count:
                    if operand.kind is not OperandKind.ADDRESS_SYMBOL:
                        raise SchemaError(
                            "active SRAM_BIND input slots require SRAM_LABEL address operands",
                            path=f"{path}.operands[{index + 1}]",
                        )
                elif (
                    operand.kind is not OperandKind.LITERAL
                    or operand.literal_value != 0
                ):
                    raise SchemaError(
                        "inactive SRAM_BIND input slots must be literal zero",
                        path=f"{path}.operands[{index + 1}]",
                    )
        if self.opcode is RecordOpcode.SRAM_ALLOC_AT:
            region_offset = self.operands[2].literal_value
            size_bytes = self.operands[3].literal_value
            alignment_bytes = self.operands[4].literal_value
            lifetime = self.operands[5].literal_value
            spillable = self.operands[6].literal_value
            if (
                type(region_offset) is not int
                or type(size_bytes) is not int
                or size_bytes == 0
                or type(alignment_bytes) is not int
                or alignment_bytes == 0
                or alignment_bytes & (alignment_bytes - 1)
                or lifetime != 0
                or type(spillable) is not bool
            ):
                raise SchemaError(
                    "SRAM_ALLOC_AT requires uint64 offset, positive size, power-of-two alignment, TASK lifetime=0 and boolean spillable",
                    path=f"{path}.operands",
                )
        if self.opcode is RecordOpcode.EVENT_WAIT:
            count = self.operands[-1].literal_value
            if type(count) is not int or count == 0 or count > 0xFFFF_FFFF:
                raise SchemaError(
                    "EVENT_WAIT count must be a non-zero uint32",
                    path=f"{path}.operands[{len(self.operands) - 1}].literal_value",
                )


@dataclass(frozen=True, slots=True)
class RuntimeRelocation:
    record_index: int
    field: RuntimeOperandField
    symbol_ref: str

    def validate(self, path: str) -> None:
        validate_uint64(self.record_index, f"{path}.record_index")
        validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")


@dataclass(frozen=True, slots=True)
class AddressRelocation:
    record_index: int
    operand_id: SemanticOperandId
    symbol_kind: ProgramSymbolKind
    symbol_ref: str
    addend: int

    def validate(self, path: str) -> None:
        validate_uint64(self.record_index, f"{path}.record_index")
        validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")
        if type(self.addend) is not int or not -(1 << 63) <= self.addend < (1 << 63):
            raise SchemaError("addend must fit signed int64", path=f"{path}.addend")


@dataclass(frozen=True, slots=True)
class CoreFragmentStream:
    logical_core: LogicalCoreRef
    records: tuple[RelocatableRecord, ...]
    runtime_relocations: tuple[RuntimeRelocation, ...]
    address_relocations: tuple[AddressRelocation, ...]

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        if not self.records:
            raise SchemaError("core fragment stream must contain records", path=f"{path}.records")
        for index, record in enumerate(self.records):
            record.validate(f"{path}.records[{index}]")
        for index, relocation in enumerate(self.runtime_relocations):
            relocation.validate(f"{path}.runtime_relocations[{index}]")
        for index, relocation in enumerate(self.address_relocations):
            relocation.validate(f"{path}.address_relocations[{index}]")

    def validate_relocations(
        self,
        runtime_symbols: dict[str, RuntimeSymbol],
        program_symbols: dict[str, ProgramSymbol],
        path: str,
    ) -> None:
        self.validate(path)

        runtime_operands: dict[tuple[int, RuntimeOperandField], RecordOperand] = {}
        address_operands: dict[tuple[int, SemanticOperandId], RecordOperand] = {}
        for record_index, record in enumerate(self.records):
            for operand in record.operands:
                if operand.kind is OperandKind.RUNTIME_SYMBOL:
                    assert operand.runtime_field is not None
                    key = (record_index, operand.runtime_field)
                    if key in runtime_operands:
                        raise SchemaError("duplicate runtime operand key", path=f"{path}.records[{record_index}].operands")
                    runtime_operands[key] = operand
                elif operand.kind is OperandKind.ADDRESS_SYMBOL:
                    assert operand.operand_id is not None
                    key = (record_index, operand.operand_id)
                    if key in address_operands:
                        raise SchemaError("duplicate address operand key", path=f"{path}.records[{record_index}].operands")
                    address_operands[key] = operand

        canonical_runtime = tuple(sorted(self.runtime_relocations, key=lambda item: (item.record_index, _RUNTIME_FIELD_ORDER[item.field])))
        if self.runtime_relocations != canonical_runtime:
            raise SchemaError("runtime relocations must be canonical", path=f"{path}.runtime_relocations")
        runtime_keys: set[tuple[int, RuntimeOperandField]] = set()
        for index, relocation in enumerate(self.runtime_relocations):
            relocation.validate(f"{path}.runtime_relocations[{index}]")
            key = (relocation.record_index, relocation.field)
            if key in runtime_keys:
                raise SchemaError("duplicate runtime relocation key", path=f"{path}.runtime_relocations[{index}]")
            runtime_keys.add(key)
            operand = runtime_operands.get(key)
            if operand is None or operand.symbol_ref != relocation.symbol_ref:
                raise SchemaError("runtime relocation does not match one runtime-symbol operand; literals take no relocation", path=f"{path}.runtime_relocations[{index}]")
            symbol = runtime_symbols.get(relocation.symbol_ref)
            if symbol is None or symbol.kind is not _RUNTIME_KIND_BY_FIELD[relocation.field]:
                raise SchemaError("runtime relocation has an unknown/wrong-kind symbol", path=f"{path}.runtime_relocations[{index}].symbol_ref")
        if runtime_keys != set(runtime_operands):
            raise SchemaError("every runtime-symbol operand requires exactly one relocation", path=f"{path}.runtime_relocations")

        canonical_address = tuple(sorted(self.address_relocations, key=lambda item: (item.record_index, int(item.operand_id))))
        if self.address_relocations != canonical_address:
            raise SchemaError("address relocations must be canonical", path=f"{path}.address_relocations")
        address_keys: set[tuple[int, SemanticOperandId]] = set()
        for index, relocation in enumerate(self.address_relocations):
            relocation.validate(f"{path}.address_relocations[{index}]")
            key = (relocation.record_index, relocation.operand_id)
            if key in address_keys:
                raise SchemaError("duplicate address relocation key", path=f"{path}.address_relocations[{index}]")
            address_keys.add(key)
            if relocation.record_index >= len(self.records):
                raise SchemaError("address relocation has a dangling record", path=f"{path}.address_relocations[{index}].record_index")
            operand = address_operands.get(key)
            if operand is None or operand.symbol_ref != relocation.symbol_ref:
                raise SchemaError("address relocation does not match one address-symbol operand; literals take no relocation", path=f"{path}.address_relocations[{index}]")
            symbol = program_symbols.get(relocation.symbol_ref)
            if symbol is None or symbol.kind is not relocation.symbol_kind:
                raise SchemaError("address relocation has an unknown/wrong-kind symbol", path=f"{path}.address_relocations[{index}].symbol_ref")
            opcode = self.records[relocation.record_index].opcode
            if relocation.symbol_kind not in _ALLOWED_ADDRESS_KINDS.get((opcode, relocation.operand_id), ()):
                raise SchemaError("program symbol kind is illegal for opcode/operand", path=f"{path}.address_relocations[{index}].symbol_kind")
            if opcode in (
                RecordOpcode.SRAM_BIND,
                RecordOpcode.SRAM_FREE,
                RecordOpcode.SRAM_ALLOC_AT,
            ) and relocation.addend != 0:
                raise SchemaError(
                    "SRAM lifecycle/bind symbolic relocation addend must be zero",
                    path=f"{path}.address_relocations[{index}].addend",
                )
        if address_keys != set(address_operands):
            raise SchemaError("every address-symbol operand requires exactly one relocation", path=f"{path}.address_relocations")


@dataclass(frozen=True, slots=True)
class BufferABI:
    id: str
    schedule_id: str
    binding_id: str
    value_id: str
    logical_core: LogicalCoreRef
    tensor_slice: TensorSlice
    region_ref: str
    region_offset_bytes: int
    size_bytes: int
    alignment_bytes: int
    banks: tuple[int, ...]
    storage_id: str
    alias_of: str | None
    lifetime_start: int
    lifetime_end_exclusive: int
    dtype: DType
    layout: str
    ownership: BufferOwnership

    def validate(self, path: str) -> None:
        for field_name in ("id", "schedule_id", "binding_id", "value_id", "region_ref", "storage_id", "layout"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        self.logical_core.validate(f"{path}.logical_core")
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.tensor_slice.value_id != self.value_id:
            raise SchemaError("tensor_slice disagrees with value_id", path=f"{path}.tensor_slice.value_id")
        for field_name in ("region_offset_bytes", "size_bytes", "alignment_bytes", "lifetime_start", "lifetime_end_exclusive"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.size_bytes == 0 or self.alignment_bytes == 0:
            raise SchemaError("size/alignment must be positive", path=path)
        if self.alignment_bytes & (self.alignment_bytes - 1):
            raise SchemaError("alignment must be a power of two", path=f"{path}.alignment_bytes")
        if self.lifetime_start >= self.lifetime_end_exclusive:
            raise SchemaError("lifetime must be a non-empty half-open interval", path=f"{path}.lifetime_end_exclusive")
        if self.alias_of is not None:
            validate_nonempty(self.alias_of, f"{path}.alias_of")
        if self.ownership is BufferOwnership.ALIASED:
            if self.alias_of is None:
                raise SchemaError("aliased buffer requires alias_of", path=f"{path}.alias_of")
        elif self.alias_of is not None:
            raise SchemaError("non-aliased buffer cannot carry alias_of", path=f"{path}.alias_of")
        if len(set(self.banks)) != len(self.banks):
            raise SchemaError("contains duplicate banks", path=f"{path}.banks")
        for index, bank in enumerate(self.banks):
            validate_uint64(bank, f"{path}.banks[{index}]")



@dataclass(frozen=True, slots=True)
class StateABI:
    """Immutable HBM state declaration carried by a command fragment."""

    id: str
    state_ref: str
    hbm_binding_ref: str
    kind: StateKind
    lifetime: PersistentStateLifetime
    access: PersistentStateAccess
    shape: tuple[int, ...]
    dtype: DType
    layout: str
    die_id: int
    address: int
    size_bytes: int
    alignment_bytes: int

    @classmethod
    def create(
        cls,
        *,
        state_ref: str,
        hbm_binding_ref: str,
        kind: StateKind,
        lifetime: PersistentStateLifetime,
        access: PersistentStateAccess,
        shape: tuple[int, ...],
        dtype: DType,
        layout: str,
        die_id: int,
        address: int,
        size_bytes: int,
        alignment_bytes: int,
    ) -> "StateABI":
        semantic_key = {
            "state_ref": state_ref,
            "hbm_binding_ref": hbm_binding_ref,
            "kind": kind,
            "lifetime": lifetime,
            "access": access,
            "shape": shape,
            "dtype": dtype,
            "layout": layout,
            "die_id": die_id,
            "address": address,
            "size_bytes": size_bytes,
            "alignment_bytes": alignment_bytes,
        }
        result = cls(
            id=stable_artifact_id(
                "state_abi",
                semantic_key,
                schema_version=STATE_ABI_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "state_ref": self.state_ref,
            "hbm_binding_ref": self.hbm_binding_ref,
            "kind": self.kind,
            "lifetime": self.lifetime,
            "access": self.access,
            "shape": self.shape,
            "dtype": self.dtype,
            "layout": self.layout,
            "die_id": self.die_id,
            "address": self.address,
            "size_bytes": self.size_bytes,
            "alignment_bytes": self.alignment_bytes,
        }

    def validate(self, path: str = "state_abi") -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.state_ref, f"{path}.state_ref")
        validate_nonempty(self.hbm_binding_ref, f"{path}.hbm_binding_ref")
        validate_nonempty(self.layout, f"{path}.layout")
        if type(self.kind) is not StateKind:
            raise SchemaError("must be a StateKind", path=f"{path}.kind")
        if type(self.lifetime) is not PersistentStateLifetime:
            raise SchemaError(
                "must be a PersistentStateLifetime", path=f"{path}.lifetime"
            )
        if type(self.access) is not PersistentStateAccess:
            raise SchemaError(
                "must be a PersistentStateAccess", path=f"{path}.access"
            )
        if type(self.shape) is not tuple or not self.shape:
            raise SchemaError(
                "must be a non-empty immutable tuple", path=f"{path}.shape"
            )
        if type(self.dtype) is not DType or self.dtype not in (DType.FP16, DType.FP32):
            raise SchemaError("unsupported state dtype", path=f"{path}.dtype")
        elements = 1
        for index, extent in enumerate(self.shape):
            validate_uint64(extent, f"{path}.shape[{index}]")
            if extent == 0:
                raise SchemaError(
                    "must be greater than zero", path=f"{path}.shape[{index}]"
                )
            if elements > ((1 << 64) - 1) // extent:
                raise SchemaError(
                    "tensor element count overflows uint64", path=f"{path}.shape"
                )
            elements *= extent
        element_bytes = 2 if self.dtype is DType.FP16 else 4
        if elements > ((1 << 64) - 1) // element_bytes:
            raise SchemaError(
                "tensor byte size overflows uint64", path=f"{path}.shape"
            )
        for field_name in ("die_id", "address", "size_bytes", "alignment_bytes"):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.die_id > (1 << 31) - 1:
            raise SchemaError("must fit signed 32-bit range", path=f"{path}.die_id")
        if self.size_bytes == 0 or self.size_bytes != elements * element_bytes:
            raise SchemaError(
                "must equal product(shape) * dtype bytes",
                path=f"{path}.size_bytes",
            )
        if (
            self.alignment_bytes == 0
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise SchemaError(
                "must be a positive power of two",
                path=f"{path}.alignment_bytes",
            )
        if self.address % self.alignment_bytes:
            raise SchemaError(
                "address must satisfy alignment", path=f"{path}.address"
            )
        if self.address > ((1 << 64) - 1) - self.size_bytes:
            raise SchemaError("HBM span overflows uint64", path=path)
        if self.kind is StateKind.PARAMETER:
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_ONLY
            ):
                raise SchemaError(
                    "parameter must be PERSISTENT and READ_ONLY", path=path
                )
        elif self.kind is StateKind.TRAINABLE_PARAMETER:
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_WRITE
            ):
                raise SchemaError(
                    "trainable parameter must be PERSISTENT and READ_WRITE",
                    path=path,
                )
        elif self.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
            if (
                self.lifetime is not PersistentStateLifetime.PERSISTENT
                or self.access is not PersistentStateAccess.READ_WRITE
            ):
                raise SchemaError(
                    "KV state must be PERSISTENT and READ_WRITE", path=path
                )
        elif self.access is not PersistentStateAccess.RESERVED:
            raise SchemaError(
                "optimizer reservation cannot grant DMA access",
                path=f"{path}.access",
            )
        expected_id = stable_artifact_id(
            "state_abi",
            self._semantic_key(),
            schema_version=STATE_ABI_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

def _fused_recv_wait_pairs(
    actions: dict[str, GlobalAction], path: str
) -> tuple[dict[str, GlobalAction], dict[str, GlobalAction]]:
    """Return exact fused/state-transfer WAIT->RECV maps frozen by N5."""

    recv_by_wait: dict[str, GlobalAction] = {}
    wait_by_recv: dict[str, GlobalAction] = {}
    for wait in actions.values():
        if wait.task_kind is not SemanticTaskKind.WAIT:
            continue
        candidates = [
            actions[dependency]
            for dependency in wait.deps
            if dependency in actions
            and actions[dependency].task_kind is SemanticTaskKind.RECV
            and actions[dependency].logical_core == wait.logical_core
            and actions[dependency].sync is not None
            and wait.sync is not None
            and actions[dependency].sync.completion_event == wait.sync.wait_event
        ]
        if len(candidates) != 1:
            raise SchemaError(
                "fused WAIT must identify exactly one same-core RECV dependency by wait event",
                path=path,
            )
        recv = candidates[0]
        fused_pair = (
            isinstance(wait.origin_ref, FusedNodeOrigin)
            and isinstance(recv.origin_ref, FusedNodeOrigin)
            and wait.origin_ref.plan_id == recv.origin_ref.plan_id
            and wait.origin_ref.rank == recv.origin_ref.rank
        )
        transfer_pair = (
            isinstance(wait.origin_ref, StateTransferOrigin)
            and isinstance(recv.origin_ref, StateTransferOrigin)
            and wait.origin_ref.state_transfer_ref
            == recv.origin_ref.state_transfer_ref
            and wait.origin_ref.rank == recv.origin_ref.rank
        )
        if (
            not (fused_pair or transfer_pair)
            or wait.runtime_binding is None
            or recv.runtime_binding is None
            or wait.runtime_binding.token_symbol is None
            or wait.runtime_binding.token_symbol != recv.runtime_binding.token_symbol
            or recv.core_order_index is None
            or wait.core_order_index is None
            or recv.core_order_index >= wait.core_order_index
        ):
            raise SchemaError(
                "WAIT must follow its same-origin/rank RECV and reuse its runtime token",
                path=path,
            )
        if recv.id in wait_by_recv:
            raise SchemaError(
                "one RECV cannot feed multiple WAIT actions",
                path=path,
            )
        recv_by_wait[wait.id] = recv
        wait_by_recv[recv.id] = wait
    return recv_by_wait, wait_by_recv


@dataclass(frozen=True, slots=True)
class _PlanBarrierRecordSpec:
    opcode: RecordOpcode
    source: GlobalAction
    destination: GlobalAction
    event: RuntimeSymbol


def _plan_barrier_group(
    dag: GlobalActionDAG, anchor: GlobalAction, path: str
) -> tuple[GlobalAction, ...]:
    plan_id, _rank, barrier, _core = _plan_barrier_action_fields(anchor, path)
    if len(barrier.participant_ranks) < 2:
        raise SchemaError(
            "coordinator PLAN barrier requires at least two participants",
            path=f"{path}.sync.barrier.participant_ranks",
        )
    candidates = tuple(
        action
        for action in dag.actions
        if action.task_kind is SemanticTaskKind.BARRIER
        and action.sync is not None
        and action.sync.barrier is not None
        and action.sync.barrier.id == barrier.id
    )
    by_rank: dict[int, GlobalAction] = {}
    for index, candidate in enumerate(candidates):
        candidate_plan, candidate_rank, candidate_barrier, _candidate_core = (
            _plan_barrier_action_fields(candidate, f"{path}.participants[{index}]")
        )
        if candidate_plan != plan_id or candidate_barrier != barrier:
            raise SchemaError(
                "shared PLAN barrier id must preserve one exact plan and contract",
                path=f"{path}.participants[{index}]",
            )
        if candidate_rank in by_rank:
            raise SchemaError(
                "PLAN barrier must contain exactly one action per participant rank",
                path=f"{path}.participants[{index}]",
            )
        by_rank[candidate_rank] = candidate
    if set(by_rank) != set(barrier.participant_ranks):
        raise SchemaError(
            "PLAN barrier actions must exactly cover participant_ranks",
            path=f"{path}.sync.barrier.participant_ranks",
        )
    ordered = tuple(by_rank[rank] for rank in barrier.participant_ranks)
    if len({action.logical_core.die_id for action in ordered}) != len(ordered):
        raise SchemaError(
            "coordinator PLAN barrier requires one distinct die per participant",
            path=f"{path}.sync.barrier.participant_ranks",
        )
    return ordered


def _plan_barrier_record_specs(
    source_global_dag_id: str,
    participants: tuple[GlobalAction, ...],
    action: GlobalAction,
) -> tuple[_PlanBarrierRecordSpec, ...]:
    leader, *peers = participants
    if action is leader:
        return (
            *(
                _PlanBarrierRecordSpec(
                    RecordOpcode.EVENT_WAIT,
                    peer,
                    leader,
                    canonical_plan_barrier_event_symbol(
                        source_global_dag_id,
                        PlanBarrierEventPhase.ARRIVE,
                        peer,
                        leader,
                    ),
                )
                for peer in peers
            ),
            *(
                _PlanBarrierRecordSpec(
                    RecordOpcode.EVENT_SET,
                    leader,
                    peer,
                    canonical_plan_barrier_event_symbol(
                        source_global_dag_id,
                        PlanBarrierEventPhase.RELEASE,
                        leader,
                        peer,
                    ),
                )
                for peer in peers
            ),
        )
    return (
        _PlanBarrierRecordSpec(
            RecordOpcode.EVENT_SET,
            action,
            leader,
            canonical_plan_barrier_event_symbol(
                source_global_dag_id,
                PlanBarrierEventPhase.ARRIVE,
                action,
                leader,
            ),
        ),
        _PlanBarrierRecordSpec(
            RecordOpcode.EVENT_WAIT,
            leader,
            action,
            canonical_plan_barrier_event_symbol(
                source_global_dag_id,
                PlanBarrierEventPhase.RELEASE,
                leader,
                action,
            ),
        ),
    )


def _plan_barrier_operands(
    source_global_dag_id: str, spec: _PlanBarrierRecordSpec
) -> tuple[RecordOperand, ...]:
    source_core = canonical_plan_barrier_core_symbol(
        source_global_dag_id, spec.source
    )
    destination_core = canonical_plan_barrier_core_symbol(
        source_global_dag_id, spec.destination
    )
    operands = (
        RecordOperand.runtime(
            "source_core", RuntimeOperandField.SOURCE_CORE, source_core.id
        ),
        RecordOperand.runtime(
            "destination_core",
            RuntimeOperandField.DESTINATION_CORE,
            destination_core.id,
        ),
        RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, spec.event.id),
    )
    if spec.opcode is RecordOpcode.EVENT_WAIT:
        return (*operands, RecordOperand.literal("count", 1))
    return operands


def _expected_plan_barrier_events(
    dag: GlobalActionDAG, path: str
) -> dict[tuple[str, str, str], _PlanBarrierRecordSpec]:
    expected: dict[tuple[str, str, str], _PlanBarrierRecordSpec] = {}
    visited: set[str] = set()
    for action in dag.actions:
        if (
            action.task_kind is not SemanticTaskKind.BARRIER
            or not isinstance(action.origin_ref, StandaloneNodeOrigin)
        ):
            continue
        participants = _plan_barrier_group(dag, action, path)
        barrier = action.sync.barrier
        assert barrier is not None
        if barrier.id in visited:
            continue
        visited.add(barrier.id)
        for participant in participants:
            for spec in _plan_barrier_record_specs(dag.id, participants, participant):
                key = (spec.source.id, spec.destination.id, spec.event.id)
                previous = expected.setdefault(key, spec)
                if (
                    previous.source != spec.source
                    or previous.destination != spec.destination
                    or previous.event != spec.event
                ):
                    raise SchemaError(
                        "canonical PLAN barrier event identity collision",
                        path=path,
                    )
    return expected


_LIFECYCLE_OPCODES = (RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE)


def _canonical_lifecycle_roots(
    buffer_abi: tuple[BufferABI, ...], *, path: str
) -> dict[str, BufferABI]:
    groups: dict[tuple[str, LogicalCoreRef, str], list[BufferABI]] = {}
    for abi in buffer_abi:
        groups.setdefault(
            (abi.schedule_id, abi.logical_core, abi.storage_id), []
        ).append(abi)
    root_by_id: dict[str, BufferABI] = {}
    for group in groups.values():
        roots = tuple(
            abi
            for abi in group
            if abi.alias_of is None
            and abi.ownership is not BufferOwnership.ALIASED
        )
        if len(roots) != 1:
            raise SchemaError(
                "lifecycle storage requires exactly one canonical non-alias root",
                path=path,
            )
        root = roots[0]
        starts = [root.lifetime_start]
        ends = [root.lifetime_end_exclusive]
        for abi in group:
            if abi is not root:
                if (
                    abi.ownership is not BufferOwnership.ALIASED
                    or abi.alias_of != root.binding_id
                    or abi.region_ref != root.region_ref
                    or abi.region_offset_bytes != root.region_offset_bytes
                    or abi.size_bytes != root.size_bytes
                    or abi.alignment_bytes != root.alignment_bytes
                    or abi.banks != root.banks
                    or abi.tensor_slice.offset != root.tensor_slice.offset
                    or abi.tensor_slice.shape != root.tensor_slice.shape
                    or abi.dtype is not root.dtype
                    or abi.layout != root.layout
                ):
                    raise SchemaError(
                        "lifecycle alias must exactly preserve its canonical root placement/view",
                        path=path,
                    )
                starts.append(abi.lifetime_start)
                ends.append(abi.lifetime_end_exclusive)
            root_by_id[abi.id] = root
        if (
            root.lifetime_start != min(starts)
            or root.lifetime_end_exclusive != max(ends)
        ):
            raise SchemaError(
                "canonical root lifetime must equal the union of all exact aliases",
                path=path,
            )
    return root_by_id


def _lifecycle_payload_indices(
    action: GlobalAction,
    records: tuple[RelocatableRecord, ...],
    indices: list[int],
    program_symbols: dict[str, ProgramSymbol],
    buffer_abi: tuple[BufferABI, ...],
    *,
    lifecycle_required: bool,
    path: str,
) -> list[int]:
    """Validate canonical action-owned fixed allocations and return payload indices."""

    first = 0
    while (
        first < len(indices)
        and records[indices[first]].opcode is RecordOpcode.SRAM_ALLOC_AT
    ):
        first += 1
    last = len(indices)
    while (
        last > first
        and records[indices[last - 1]].opcode is RecordOpcode.SRAM_FREE
    ):
        last -= 1
    payload = indices[first:last]
    if any(records[index].opcode in _LIFECYCLE_OPCODES for index in payload):
        raise SchemaError(
            "SRAM_ALLOC_AT must be an action prefix and SRAM_FREE an action suffix",
            path=path,
        )
    if not lifecycle_required:
        return payload
    if action.core_order_index is None:
        raise SchemaError("lifecycle action lacks core order", path=path)

    abi_by_binding = {
        (abi.schedule_id, abi.binding_id): abi for abi in buffer_abi
    }
    root_by_id = _canonical_lifecycle_roots(buffer_abi, path=path)
    used: dict[str, BufferABI] = {}
    for use in action.buffer_uses:
        abi = abi_by_binding.get((action.source.schedule_id, use.binding_id))
        if abi is None:
            raise SchemaError(
                "lifecycle action use lacks its BufferABI",
                path=path,
            )
        root = root_by_id[abi.id]
        previous = used.setdefault(root.storage_id, root)
        if previous != root:
            raise SchemaError(
                "lifecycle action uses conflicting canonical roots for one storage",
                path=path,
            )

    def order_key(abi: BufferABI) -> tuple[str, int, str, str]:
        return (abi.region_ref, abi.region_offset_bytes, abi.storage_id, abi.id)

    expected_allocs = tuple(
        sorted(
            (
                abi
                for abi in used.values()
                if abi.lifetime_start == action.core_order_index
            ),
            key=order_key,
        )
    )
    expected_frees = tuple(
        reversed(
            sorted(
                (
                    abi
                    for abi in used.values()
                    if abi.lifetime_end_exclusive
                    == action.core_order_index + 1
                ),
                key=order_key,
            )
        )
    )
    actual_allocs = tuple(records[index] for index in indices[:first])
    actual_frees = tuple(records[index] for index in indices[last:])
    if len(actual_allocs) != len(expected_allocs) or len(actual_frees) != len(
        expected_frees
    ):
        raise SchemaError(
            "lifecycle records must exactly cover storage first/last uses",
            path=path,
        )

    for record, abi in zip(actual_allocs, expected_allocs):
        region_symbol = program_symbols[record.operands[0].symbol_ref]
        label_symbol = program_symbols[record.operands[1].symbol_ref]
        if (
            region_symbol.kind is not ProgramSymbolKind.SRAM_REGION
            or region_symbol.source_ref != abi.region_ref
            or label_symbol.kind is not ProgramSymbolKind.SRAM_LABEL
            or label_symbol.source_ref != abi.storage_id
            or tuple(operand.literal_value for operand in record.operands[2:6])
            != (
                abi.region_offset_bytes,
                abi.size_bytes,
                abi.alignment_bytes,
                0,
            )
            or type(record.operands[6].literal_value) is not bool
        ):
            raise SchemaError(
                "SRAM_ALLOC_AT must exactly preserve BufferABI placement and storage identity",
                path=path,
            )
    for record, abi in zip(actual_frees, expected_frees):
        label_symbol = program_symbols[record.operands[0].symbol_ref]
        if (
            label_symbol.kind is not ProgramSymbolKind.SRAM_LABEL
            or label_symbol.source_ref != abi.storage_id
        ):
            raise SchemaError(
                "SRAM_FREE must exactly identify the ending BufferABI storage",
                path=path,
            )
    return payload


@dataclass(frozen=True, slots=True)
class CommandFragment:
    schema_version: str
    producer_pass: str
    id: str
    source_global_dag_id: str
    kind: FragmentKind
    claimed_action_ids: tuple[str, ...]
    core_streams: tuple[CoreFragmentStream, ...]
    runtime_symbols: tuple[RuntimeSymbol, ...]
    program_symbols: tuple[ProgramSymbol, ...]
    buffer_abi: tuple[BufferABI, ...]
    state_abi: tuple[StateABI, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "CommandFragment":
        semantic_key.setdefault("state_abi", ())
        return cls(
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("command_fragment", semantic_key, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_global_dag_id", "kind", "claimed_action_ids", "core_streams",
            "runtime_symbols", "program_symbols", "buffer_abi",
            "state_abi",
        )}

    def validate(self, path: str = "command_fragment") -> None:
        if self.schema_version != COMMAND_FRAGMENT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_global_dag_id, f"{path}.source_global_dag_id")
        if not self.claimed_action_ids or self.claimed_action_ids != tuple(sorted(set(self.claimed_action_ids))):
            raise SchemaError("claimed action ids must be non-empty, unique and canonical", path=f"{path}.claimed_action_ids")
        if not self.core_streams:
            raise SchemaError("must contain core streams", path=f"{path}.core_streams")
        if tuple(stream.logical_core for stream in self.core_streams) != tuple(sorted((stream.logical_core for stream in self.core_streams), key=lambda core: (core.die_id, core.local_core_id))):
            raise SchemaError("core streams must be in canonical logical-core order", path=f"{path}.core_streams")
        runtime_symbols = {symbol.id: symbol for symbol in self.runtime_symbols}
        program_symbols = {symbol.id: symbol for symbol in self.program_symbols}
        if len(runtime_symbols) != len(self.runtime_symbols) or tuple(runtime_symbols) != tuple(sorted(runtime_symbols)):
            raise SchemaError("runtime symbols must have unique canonical ids", path=f"{path}.runtime_symbols")
        if len(program_symbols) != len(self.program_symbols) or tuple(program_symbols) != tuple(sorted(program_symbols)):
            raise SchemaError("program symbols must have unique canonical ids", path=f"{path}.program_symbols")
        for index, symbol in enumerate(self.runtime_symbols):
            symbol.validate(f"{path}.runtime_symbols[{index}]")
        for index, symbol in enumerate(self.program_symbols):
            symbol.validate(f"{path}.program_symbols[{index}]")
        record_actions: dict[str, tuple[int, list[int]]] = {}
        stream_cores: set[LogicalCoreRef] = set()
        for stream_index, stream in enumerate(self.core_streams):
            if stream.logical_core in stream_cores:
                raise SchemaError("duplicate logical core stream", path=f"{path}.core_streams[{stream_index}].logical_core")
            stream_cores.add(stream.logical_core)
            stream.validate_relocations(runtime_symbols, program_symbols, f"{path}.core_streams[{stream_index}]")
            for record_index, record in enumerate(stream.records):
                if record.source_global_action_id not in self.claimed_action_ids:
                    raise SchemaError("record origin is not claimed by the fragment", path=f"{path}.core_streams[{stream_index}].records[{record_index}].source_global_action_id")
                previous = record_actions.get(record.source_global_action_id)
                if previous is not None and previous[0] != stream_index:
                    raise SchemaError("one action cannot emit records on multiple streams", path=f"{path}.core_streams[{stream_index}].records[{record_index}]")
                record_actions.setdefault(record.source_global_action_id, (stream_index, []))[1].append(record_index)
        if set(record_actions) != set(self.claimed_action_ids):
            raise SchemaError("every claimed action must emit at least one record", path=f"{path}.claimed_action_ids")
        for action_id, (_stream, indices) in record_actions.items():
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise SchemaError(f"action {action_id!r} records must be contiguous", path=f"{path}.core_streams")
        if tuple(binding.id for binding in self.buffer_abi) != tuple(sorted({binding.id for binding in self.buffer_abi})):
            raise SchemaError("buffer ABI ids must be unique and canonical", path=f"{path}.buffer_abi")
        for index, binding in enumerate(self.buffer_abi):
            binding.validate(f"{path}.buffer_abi[{index}]")
            if binding.logical_core not in stream_cores:
                raise SchemaError("buffer ABI references a non-target core", path=f"{path}.buffer_abi[{index}].logical_core")
        state_ids = tuple(abi.id for abi in self.state_abi)
        if state_ids != tuple(sorted(set(state_ids))):
            raise SchemaError(
                "state ABI ids must be unique and canonical",
                path=f"{path}.state_abi",
            )
        state_by_hbm_ref: dict[str, StateABI] = {}
        for index, abi in enumerate(self.state_abi):
            abi.validate(f"{path}.state_abi[{index}]")
            if abi.hbm_binding_ref in state_by_hbm_ref:
                raise SchemaError(
                    "one fragment cannot declare multiple StateABI values for one HBM binding",
                    path=f"{path}.state_abi[{index}].hbm_binding_ref",
                )
            state_by_hbm_ref[abi.hbm_binding_ref] = abi
        if (self.kind is FragmentKind.STATE_IO) != bool(self.state_abi):
            raise SchemaError(
                "STATE_IO fragments require StateABI and all other fragments forbid it",
                path=f"{path}.state_abi",
            )
        rooted_ar = self.kind is FragmentKind.S2_LITE_ROOTED_AR
        if rooted_ar != (self.producer_pass == "s2_lite_rooted_ar_lowering"):
            raise SchemaError(
                "S2_LITE_ROOTED_AR kind is reserved for its exact dedicated producer",
                path=f"{path}.kind",
            )
        if rooted_ar:
            allowed = {
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.SRAM_FREE,
                RecordOpcode.DTE_ISSUE,
                RecordOpcode.DTE_SEND,
                RecordOpcode.DTE_RECV,
                RecordOpcode.DTE_WAIT,
                RecordOpcode.LOCAL_REDUCE,
            }
            if any(
                record.opcode not in allowed
                for stream in self.core_streams
                for record in stream.records
            ):
                raise SchemaError(
                    "rooted-AR fragments only permit scratch/DTE/reduce records",
                    path=f"{path}.core_streams",
                )
        witnessed_state_ids: set[str] = set()
        for stream_index, stream in enumerate(self.core_streams):
            for relocation_index, relocation in enumerate(stream.address_relocations):
                if relocation.operand_id is not SemanticOperandId.HBM_ADDRESS:
                    continue
                record = stream.records[relocation.record_index]
                relocation_path = (
                    f"{path}.core_streams[{stream_index}]"
                    f".address_relocations[{relocation_index}]"
                )
                if (
                    record.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE)
                    or relocation.symbol_kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                ):
                    raise SchemaError(
                        "HBM relocation requires one blocking LSU absolute address",
                        path=relocation_path,
                    )
                symbol = program_symbols[relocation.symbol_ref]
                abi = state_by_hbm_ref.get(symbol.source_ref)
                if abi is None:
                    raise SchemaError(
                        "HBM relocation has no matching StateABI",
                        path=f"{relocation_path}.symbol_ref",
                    )
                if stream.logical_core.die_id != abi.die_id:
                    raise SchemaError(
                        "HBM relocation executes outside the StateABI home die",
                        path=relocation_path,
                    )
                size_bytes = record.operands[1].literal_value
                assert size_bytes is not None
                if (
                    relocation.addend > abi.size_bytes
                    or size_bytes > abi.size_bytes - relocation.addend
                ):
                    raise SchemaError(
                        "LSU byte range must be contained in StateABI",
                        path=relocation_path,
                    )
                if (
                    record.opcode is RecordOpcode.LSU_LOAD
                    and abi.access is PersistentStateAccess.RESERVED
                ) or (
                    record.opcode is RecordOpcode.LSU_STORE
                    and abi.access is not PersistentStateAccess.READ_WRITE
                ):
                    raise SchemaError(
                        "LSU direction is forbidden by StateABI access",
                        path=relocation_path,
                    )
                witnessed_state_ids.add(abi.id)
        if witnessed_state_ids != set(state_ids):
            raise SchemaError(
                "every StateABI requires at least one exact HBM relocation witness",
                path=f"{path}.state_abi",
            )
        expected_id = stable_artifact_id("command_fragment", self._semantic_key(), schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against_lite_moe_intent(
        self,
        intent: "LiteMoeN6Intent",
        global_dag: "LiteMoeGlobalDag",
        schedule: "LiteMoeScheduled",
        source: "LiteMoeN4IR1",
        path: str = "command_fragment",
    ) -> None:
        """Validate one fragment against the exact dedicated MoE intent."""

        from ..lowering.lite_moe import validate_lite_moe_fragment

        validate_lite_moe_fragment(
            self, intent, global_dag, schedule, source, path
        )

    def validate_against(self, dag: GlobalActionDAG, path: str = "command_fragment") -> None:
        self.validate(path)
        dag.validate("global_action_dag")
        if self.source_global_dag_id != dag.id:
            raise SchemaError("fragment references a different global DAG", path=f"{path}.source_global_dag_id")
        actions = {action.id: action for action in dag.actions}
        wave_incoming_by_send, wave_outgoing_by_wait = (
            state_transfer_wave_action_maps(actions, path)
        )
        recv_by_wait, wait_by_recv = _fused_recv_wait_pairs(actions, path)
        runtime_symbols = {symbol.id: symbol for symbol in self.runtime_symbols}
        program_symbols = {symbol.id: symbol for symbol in self.program_symbols}
        if not set(self.claimed_action_ids).issubset(actions):
            raise SchemaError("fragment claims a dangling global action", path=f"{path}.claimed_action_ids")
        records_by_action: dict[str, tuple[int, list[int], LogicalCoreRef]] = {}
        for stream_index, stream in enumerate(self.core_streams):
            for record_index, record in enumerate(stream.records):
                entry = records_by_action.setdefault(record.source_global_action_id, (stream_index, [], stream.logical_core))
                entry[1].append(record_index)
        if self.kind is FragmentKind.STATE_TRANSFER:
            claimed = tuple(actions[action_id] for action_id in self.claimed_action_ids)
            if any(
                not isinstance(action.origin_ref, StateTransferOrigin)
                for action in claimed
            ):
                raise SchemaError(
                    "STATE_TRANSFER may claim only StateTransferOrigin actions",
                    path=f"{path}.claimed_action_ids",
                )
            transfer_refs = {
                action.origin_ref.state_transfer_ref for action in claimed
            }
            logical_cores = {action.logical_core for action in claimed}
            if (
                len(transfer_refs) != 1
                or len(logical_cores) != 1
                or None in logical_cores
                or len(self.core_streams) != 1
                or self.core_streams[0].logical_core
                != next(iter(logical_cores))
            ):
                raise SchemaError(
                    "STATE_TRANSFER requires one contract on one executable endpoint core",
                    path=f"{path}.claimed_action_ids",
                )
            transfer_ref = next(iter(transfer_refs))
            logical_core = next(iter(logical_cores))
            assert logical_core is not None
            expected_action_ids = tuple(
                sorted(
                    action.id
                    for action in dag.actions
                    if isinstance(action.origin_ref, StateTransferOrigin)
                    and action.origin_ref.state_transfer_ref == transfer_ref
                    and action.logical_core is not None
                    and action.logical_core.die_id == logical_core.die_id
                )
            )
            if self.claimed_action_ids != expected_action_ids:
                raise SchemaError(
                    "STATE_TRANSFER must exactly claim one contract/die endpoint action set",
                    path=f"{path}.claimed_action_ids",
                )
            ordered_claimed = tuple(
                sorted(claimed, key=lambda action: action.core_order_index)
            )
            kinds = tuple(action.task_kind for action in ordered_claimed)
            origins = tuple(
                action.origin_ref for action in ordered_claimed
            )
            segment_indices = tuple(
                origin.segment_index for origin in origins
            )
            legacy = all(index is None for index in segment_indices)
            source_endpoint = (
                kinds == (SemanticTaskKind.SEND,)
                and ordered_claimed[0].flow_route is not None
                and ordered_claimed[0].flow_route.role is FlowRouteRole.SOURCE
            ) if legacy else (
                all(kind is SemanticTaskKind.SEND for kind in kinds)
                and segment_indices == tuple(range(len(ordered_claimed)))
                and all(
                    action.flow_route is not None
                    and action.flow_route.role is FlowRouteRole.SOURCE
                    for action in ordered_claimed
                )
            )
            destination_endpoint = (
                kinds == (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
                and ordered_claimed[0].flow_route is not None
                and ordered_claimed[0].flow_route.role
                is FlowRouteRole.DESTINATION
                and ordered_claimed[1].flow_route is None
            ) if legacy else (
                len(ordered_claimed) % 2 == 0
                and kinds
                == tuple(
                    kind
                    for _ in range(len(ordered_claimed) // 2)
                    for kind in (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
                )
                and segment_indices
                == tuple(
                    segment_index
                    for segment_index in range(len(ordered_claimed) // 2)
                    for _ in range(2)
                )
                and all(
                    recv.flow_route is not None
                    and recv.flow_route.role is FlowRouteRole.DESTINATION
                    and wait.flow_route is None
                    for recv, wait in zip(
                        ordered_claimed[::2], ordered_claimed[1::2]
                    )
                )
            )
            if not (source_endpoint or destination_endpoint):
                raise SchemaError(
                    "STATE_TRANSFER endpoint must be legacy SEND/RECV+WAIT or canonical contiguous segmented SEND/RECV+WAIT",
                    path=f"{path}.claimed_action_ids",
                )
            validate_state_transfer_wave_fragment_shape(
                ordered_claimed,
                wave_incoming_by_send,
                wave_outgoing_by_wait,
                f"{path}.core_streams[0].records",
            )
            expected_buffer_keys = tuple(
                sorted(
                    {
                        (action.source.schedule_id, use.binding_id)
                        for action in claimed
                        for use in action.buffer_uses
                    }
                )
            )
            actual_buffer_keys = tuple(
                sorted(
                    (abi.schedule_id, abi.binding_id)
                    for abi in self.buffer_abi
                )
            )
            if actual_buffer_keys != expected_buffer_keys:
                raise SchemaError(
                    "STATE_TRANSFER BufferABI must exactly cover its local endpoint uses",
                    path=f"{path}.buffer_abi",
                )
            used_runtime_symbols = {
                relocation.symbol_ref
                for stream in self.core_streams
                for relocation in stream.runtime_relocations
            }
            used_program_symbols = {
                relocation.symbol_ref
                for stream in self.core_streams
                for relocation in stream.address_relocations
            }
            if used_runtime_symbols != set(runtime_symbols):
                raise SchemaError(
                    "STATE_TRANSFER runtime symbols must have exact relocation witnesses",
                    path=f"{path}.runtime_symbols",
                )
            if used_program_symbols != set(program_symbols):
                raise SchemaError(
                    "STATE_TRANSFER program symbols must have exact relocation witnesses",
                    path=f"{path}.program_symbols",
                )
        allowed_opcodes = {
            SemanticTaskKind.SEND: (RecordOpcode.DTE_SEND,),
            SemanticTaskKind.RECV: (RecordOpcode.DTE_RECV,),
            SemanticTaskKind.REDUCE: (RecordOpcode.LOCAL_REDUCE,),
            SemanticTaskKind.LOCAL_COPY: (RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT),
            SemanticTaskKind.WAIT: (RecordOpcode.DTE_WAIT,),
            SemanticTaskKind.BARRIER: (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT),
            SemanticTaskKind.DMA_IN: (RecordOpcode.LSU_LOAD,),
            SemanticTaskKind.DMA_OUT: (RecordOpcode.LSU_STORE,),
        }
        lifecycle_required = any(
            record.opcode in _LIFECYCLE_OPCODES
            for stream in self.core_streams
            for record in stream.records
        )
        sequence_by_stream: dict[int, list[str]] = {}
        expected_plan_runtime_symbols: dict[str, RuntimeSymbol] = {}
        claimed_plan_barrier_ids: set[str] = set()
        for action_id in self.claimed_action_ids:
            action = actions[action_id]
            if action.task_kind is SemanticTaskKind.TRANSIT:
                raise SchemaError("TRANSIT cannot be claimed or emitted as a record", path=f"{path}.claimed_action_ids")
            if action.logical_core is None:
                raise SchemaError("claimed executable action lacks a core", path=f"{path}.claimed_action_ids")
            if self.kind is not _FRAGMENT_KIND_BY_LOWERING[action.lowering]:
                raise SchemaError("fragment kind disagrees with action RegionLowering", path=f"{path}.kind")
            stream_index, indices, core = records_by_action[action_id]
            if core != action.logical_core:
                raise SchemaError("record stream core disagrees with action core", path=f"{path}.core_streams[{stream_index}].logical_core")
            records = self.core_streams[stream_index].records
            indices = _lifecycle_payload_indices(
                action,
                records,
                indices,
                {symbol.id: symbol for symbol in self.program_symbols},
                self.buffer_abi,
                lifecycle_required=lifecycle_required,
                path=f"{path}.core_streams[{stream_index}].records",
            )
            if not indices:
                raise SchemaError(
                    "every action requires at least one non-lifecycle payload record",
                    path=f"{path}.core_streams[{stream_index}].records",
                )
            if self.kind is FragmentKind.STATE_TRANSFER:
                incoming = wave_incoming_by_send.get(action.id)
                if incoming is not None:
                    expected_wait = canonical_state_transfer_wave_record(
                        dag.id,
                        owner=action,
                        source=incoming,
                        destination=action,
                        opcode=RecordOpcode.EVENT_WAIT,
                    )
                    if records[indices[0]] != expected_wait:
                        raise SchemaError(
                            "segmented wave SEND requires one exact leading EVENT_WAIT",
                            path=f"{path}.core_streams[{stream_index}].records",
                        )
                    indices = indices[1:]
                outgoing = wave_outgoing_by_wait.get(action.id)
                if outgoing is not None:
                    expected_set = canonical_state_transfer_wave_record(
                        dag.id,
                        owner=action,
                        source=action,
                        destination=outgoing,
                        opcode=RecordOpcode.EVENT_SET,
                    )
                    if (
                        not indices
                        or records[indices[-1]] != expected_set
                    ):
                        raise SchemaError(
                            "segmented wave WAIT requires one exact trailing EVENT_SET",
                            path=f"{path}.core_streams[{stream_index}].records",
                        )
                    indices = indices[:-1]
                if not indices:
                    raise SchemaError(
                        "wave EVENT cannot replace the transfer payload",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            if action.task_kind is SemanticTaskKind.COMP:
                assert action.compute is not None
                abi = _compute_record_abi(
                    action.compute,
                    path=f"{path}.actions[{action.id}].compute",
                )
                if (
                    len(indices) != 2
                    or indices[1] != indices[0] + 1
                    or records[indices[0]].opcode is not RecordOpcode.SRAM_BIND
                    or records[indices[1]].opcode is not abi.opcode
                ):
                    raise SchemaError(
                        "COMP requires exact contiguous SRAM_BIND then compute mapping",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                bind = records[indices[0]]
                if (
                    bind.operands[0].literal_value != abi.bind_input_count
                ):
                    raise SchemaError(
                        "compute SRAM_BIND input_count must exactly match its frozen public input ABI",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                input_uses = tuple(
                    sorted(
                        (
                            use
                            for use in action.buffer_uses
                            if use.role is BufferUseRole.COMP_INPUT
                        ),
                        key=lambda use: use.operand_index,
                    )
                )
                output_uses = tuple(
                    use
                    for use in action.buffer_uses
                    if use.role is BufferUseRole.COMP_OUTPUT
                )
                if (
                    tuple(use.operand_index for use in input_uses)
                    != tuple(range(len(action.compute.inputs)))
                    or len(output_uses) != 1
                    or output_uses[0].operand_index != 0
                    or len(action.buffer_uses)
                    != len(action.compute.inputs) + len(action.compute.outputs)
                ):
                    raise SchemaError(
                        "compute requires exact ordered input/output BufferABI roles",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                compute_record = records[indices[1]]
                if abi.opcode in _FIXED_COMPUTE_OPCODES:
                    expected_literals = _fixed_compute_literals(
                        action.compute,
                        abi.opcode,
                        path=f"{path}.actions[{action.id}].compute",
                    )
                    actual_literals = {
                        operand.name: operand.literal_value
                        for operand in compute_record.operands
                        if operand.kind is OperandKind.LITERAL
                    }
                    if actual_literals != expected_literals:
                        raise SchemaError(
                            "fixed compute record must exactly derive from its typed workload",
                            path=f"{path}.core_streams[{stream_index}].records",
                        )
                elif compute_record.operands[-1].literal_value != abi.parameters:
                    raise SchemaError(
                        "compute parameters must exactly derive from the typed rank-local workload",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            elif action.task_kind in (
                SemanticTaskKind.DMA_IN,
                SemanticTaskKind.DMA_OUT,
            ):
                expected_opcode = (
                    RecordOpcode.LSU_LOAD
                    if action.task_kind is SemanticTaskKind.DMA_IN
                    else RecordOpcode.LSU_STORE
                )
                if (
                    len(indices) != 1
                    or records[indices[0]].opcode is not expected_opcode
                    or action.dma is None
                    or len(action.state_uses) != 1
                    or len(action.buffer_uses) != 1
                ):
                    raise SchemaError(
                        "state DMA requires one exact direction-matching blocking LSU record",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                record = records[indices[0]]
                state_use = action.state_uses[0]
                matching_state = tuple(
                    abi
                    for abi in self.state_abi
                    if abi.hbm_binding_ref == state_use.hbm_binding_ref
                )
                local_use = action.buffer_uses[0]
                matching_local = tuple(
                    abi
                    for abi in self.buffer_abi
                    if (
                        abi.schedule_id,
                        abi.binding_id,
                    )
                    == (
                        action.source.schedule_id,
                        local_use.binding_id,
                    )
                )
                if len(matching_state) != 1 or len(matching_local) != 1:
                    raise SchemaError(
                        "state DMA requires exact StateABI and local BufferABI endpoints",
                        path=f"{path}.state_abi",
                    )
                state = matching_state[0]
                local = matching_local[0]
                if (
                    state.state_ref != action.dma.state_ref
                    or state.die_id != action.logical_core.die_id
                    or state.dtype is not action.dtype
                    or state.layout != local.layout
                    or action.bytes > local.size_bytes
                    or record.operands[1].literal_value != action.bytes
                ):
                    raise SchemaError(
                        "state DMA record/StateABI does not exactly preserve GlobalAction semantics",
                        path=f"{path}.state_abi",
                    )
                if (
                    action.dma.state_offset_bytes > state.size_bytes
                    or action.bytes
                    > state.size_bytes - action.dma.state_offset_bytes
                ):
                    raise SchemaError(
                        "state DMA action byte range exceeds StateABI",
                        path=f"{path}.state_abi",
                    )
            elif action.task_kind is SemanticTaskKind.LOCAL_COPY:
                if tuple(records[index].opcode for index in indices) != (
                    RecordOpcode.DTE_ISSUE,
                    RecordOpcode.DTE_WAIT,
                ):
                    raise SchemaError("LOCAL_COPY requires exact contiguous DTE_ISSUE then DTE_WAIT pattern", path=f"{path}.core_streams[{stream_index}].records")
            elif action.task_kind is SemanticTaskKind.BARRIER:
                participants = _plan_barrier_group(dag, action, path)
                participant_ids = {participant.id for participant in participants}
                if not participant_ids.issubset(self.claimed_action_ids):
                    raise SchemaError(
                        "one coordinator PLAN barrier and all participant actions must be emitted by one fragment",
                        path=f"{path}.claimed_action_ids",
                    )
                barrier = action.sync.barrier
                assert barrier is not None
                claimed_plan_barrier_ids.add(barrier.id)
                for participant in participants:
                    symbol = canonical_plan_barrier_core_symbol(dag.id, participant)
                    expected_plan_runtime_symbols[symbol.id] = symbol
                    for spec in _plan_barrier_record_specs(
                        dag.id, participants, participant
                    ):
                        expected_plan_runtime_symbols[spec.event.id] = spec.event
                specs = _plan_barrier_record_specs(dag.id, participants, action)
                actual_records = tuple(records[index] for index in indices)
                expected_records = tuple(
                    RelocatableRecord(
                        action.id,
                        spec.opcode,
                        _plan_barrier_operands(dag.id, spec),
                    )
                    for spec in specs
                )
                if actual_records != expected_records:
                    raise SchemaError(
                        "PLAN barrier requires the exact participant-ordered coordinator ARRIVE/RELEASE record sequence",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            elif (
                action.task_kind not in allowed_opcodes
                or len(indices) != 1
                or records[indices[0]].opcode not in allowed_opcodes[action.task_kind]
            ):
                raise SchemaError("opcode is incompatible with source task kind", path=f"{path}.core_streams[{stream_index}].records")
            if action.task_kind is SemanticTaskKind.REDUCE:
                reduction = action.reduction
                if reduction is None:
                    raise SchemaError(
                        "LOCAL_REDUCE requires the source action reduction contract",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                operands = {
                    operand.name: operand
                    for operand in records[indices[0]].operands
                }
                fp32_dp2 = (
                    reduction.input_dtype is DType.FP32
                    and reduction.accumulation_dtype is DType.FP32
                    and reduction.output_dtype is DType.FP32
                    and reduction.input_ranks == (0, 1)
                    and action.dtype is DType.FP32
                    and action.bytes == 2048
                )
                dtype_literal = 1 if fp32_dp2 else 0
                if any(
                    operands[name].literal_value != expected
                    for name, expected in (
                        ("input_dtype", dtype_literal),
                        ("accumulator_dtype", 1),
                        ("output_dtype", dtype_literal),
                        ("reduce_op", 1),
                        ("rounding", 0),
                        ("order", 0),
                    )
                ):
                    raise SchemaError(
                        "LOCAL_REDUCE literals do not match fixed FP16/FP32/FP16 or exact DP2 FP32/FP32/FP32",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                input_count = operands["input_count"].literal_value
                if (
                    input_count == 0
                    or input_count > (1 << 16) - 1
                    or input_count != len(reduction.input_ranks)
                ):
                    raise SchemaError(
                        "LOCAL_REDUCE input_count must be non-zero u16 and exactly match reduction input_ranks",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                element_count = operands["element_count"].literal_value
                element_bytes = 4 if fp32_dp2 else 2
                if (
                    action.bytes == 0
                    or action.bytes % element_bytes
                    or element_count == 0
                    or element_count != action.bytes // element_bytes
                ):
                    raise SchemaError(
                        "LOCAL_REDUCE requires positive even FP16 bytes or exact 4-byte-aligned DP2 FP32 bytes and exact non-zero element_count",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
                if operands["input_stride_bytes"].literal_value != action.bytes:
                    raise SchemaError(
                        "LOCAL_REDUCE input_stride_bytes must exactly equal action bytes",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            if action.task_kind in (SemanticTaskKind.SEND, SemanticTaskKind.RECV):
                operands = {operand.name: operand for operand in records[indices[0]].operands}
                zero_fields = (
                    "mode",
                    "datatype",
                    "reduce_op",
                    "expected_sources",
                    "tree_id",
                    "group_id",
                    "collective_id",
                    "epoch",
                )
                if action.task_kind is SemanticTaskKind.SEND:
                    zero_fields = (*zero_fields, "source_space")
                expected_async_wait = wait_by_recv.get(action.id)
                transport_completion_is_exact = (
                    operands["completion"].literal_value == 1
                    and operands["token"].kind is OperandKind.LITERAL
                    and operands["token"].literal_value == 0
                )
                if action.task_kind is SemanticTaskKind.RECV and expected_async_wait is not None:
                    wait_stream_index, wait_indices, wait_core = records_by_action.get(
                        expected_async_wait.id, (-1, [], LogicalCoreRef(-1, -1))
                    )
                    recv_token = operands["token"]
                    wait_records = (
                        self.core_streams[wait_stream_index].records
                        if wait_stream_index >= 0
                        else ()
                    )
                    if wait_records:
                        wait_indices = _lifecycle_payload_indices(
                            expected_async_wait,
                            wait_records,
                            wait_indices,
                            {symbol.id: symbol for symbol in self.program_symbols},
                            self.buffer_abi,
                            lifecycle_required=lifecycle_required,
                            path=f"{path}.core_streams[{wait_stream_index}].records",
                        )
                        outgoing = wave_outgoing_by_wait.get(
                            expected_async_wait.id
                        )
                        if outgoing is not None:
                            expected_set = (
                                canonical_state_transfer_wave_record(
                                    dag.id,
                                    owner=expected_async_wait,
                                    source=expected_async_wait,
                                    destination=outgoing,
                                    opcode=RecordOpcode.EVENT_SET,
                                )
                            )
                            if (
                                not wait_indices
                                or wait_records[wait_indices[-1]]
                                != expected_set
                            ):
                                raise SchemaError(
                                    "segmented wave WAIT requires one exact trailing EVENT_SET",
                                    path=f"{path}.core_streams[{wait_stream_index}].records",
                                )
                            wait_indices = wait_indices[:-1]
                    wait_token = (
                        wait_records[wait_indices[0]].operands[0]
                        if len(wait_indices) == 1 and wait_records
                        else None
                    )
                    token_symbol = (
                        runtime_symbols.get(recv_token.symbol_ref or "")
                        if recv_token.kind is OperandKind.RUNTIME_SYMBOL
                        else None
                    )
                    transport_completion_is_exact = (
                        operands["completion"].literal_value == 0
                        and recv_token.kind is OperandKind.RUNTIME_SYMBOL
                        and wait_stream_index == stream_index
                        and wait_core == core
                        and len(wait_indices) == 1
                        and wait_token is not None
                        and wait_token.kind is OperandKind.RUNTIME_SYMBOL
                        and recv_token.symbol_ref == wait_token.symbol_ref
                        and token_symbol is not None
                        and token_symbol.kind is RuntimeSymbolKind.DTE_TOKEN
                        and action.runtime_binding is not None
                        and expected_async_wait.runtime_binding is not None
                        and token_symbol.source_ref
                        == action.runtime_binding.token_symbol
                        == expected_async_wait.runtime_binding.token_symbol
                    )
                if (
                    operands["length_bytes"].literal_value != action.bytes
                    or action.bytes == 0
                    or action.bytes > _DTE_ENDPOINT_P2P_MAX_BYTES
                    or not transport_completion_is_exact
                    or any(
                        operands[field].kind is not OperandKind.LITERAL
                        or operands[field].literal_value != 0
                        for field in zero_fields
                    )
                ):
                    raise SchemaError(
                        "P2P transport requires canonical zero literals and bounded exact bytes; SEND/standalone RECV use SYNC token=0 while waited fused RECV uses ASYNC shared DTE token",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            if action.task_kind is SemanticTaskKind.WAIT:
                recv = recv_by_wait[action.id]
                recv_stream_index, recv_indices, recv_core = records_by_action.get(
                    recv.id, (-1, [], LogicalCoreRef(-1, -1))
                )
                wait_token = records[indices[0]].operands[0]
                recv_records = (
                    self.core_streams[recv_stream_index].records
                    if recv_stream_index >= 0
                    else ()
                )
                if recv_records:
                    recv_indices = _lifecycle_payload_indices(
                        recv,
                        recv_records,
                        recv_indices,
                        {symbol.id: symbol for symbol in self.program_symbols},
                        self.buffer_abi,
                        lifecycle_required=lifecycle_required,
                        path=f"{path}.core_streams[{recv_stream_index}].records",
                    )
                recv_token = (
                    recv_records[recv_indices[0]].operands[5]
                    if len(recv_indices) == 1 and recv_records
                    else None
                )
                token_symbol = (
                    runtime_symbols.get(wait_token.symbol_ref or "")
                    if wait_token.kind is OperandKind.RUNTIME_SYMBOL
                    else None
                )
                if (
                    recv_stream_index != stream_index
                    or recv_core != core
                    or len(recv_indices) != 1
                    or recv_token is None
                    or recv_token.kind is not OperandKind.RUNTIME_SYMBOL
                    or wait_token.kind is not OperandKind.RUNTIME_SYMBOL
                    or recv_token.symbol_ref != wait_token.symbol_ref
                    or token_symbol is None
                    or token_symbol.kind is not RuntimeSymbolKind.DTE_TOKEN
                    or recv.runtime_binding is None
                    or action.runtime_binding is None
                    or token_symbol.source_ref
                    != recv.runtime_binding.token_symbol
                    or token_symbol.source_ref
                    != action.runtime_binding.token_symbol
                ):
                    raise SchemaError(
                        "fused WAIT must emit one DTE_WAIT sharing its same-core waited RECV runtime token",
                        path=f"{path}.core_streams[{stream_index}].records",
                    )
            if action.task_kind is SemanticTaskKind.LOCAL_COPY:
                operands = {operand.name: operand for operand in records[indices[0]].operands}
                wait_operands = {operand.name: operand for operand in records[indices[1]].operands}
                if (
                    operands["direction"].literal_value != 0
                    or action.bytes > ((1 << 64) - 1) // 8
                    or operands["payload_bits"].literal_value != action.bytes * 8
                    or operands["size_bytes"].literal_value != action.bytes
                    or operands["hbm_address"].literal_value != 0
                    or operands["token"].symbol_ref != wait_operands["token"].symbol_ref
                ):
                    raise SchemaError("LOCAL_COPY requires exact SPM_TO_SPM literals and one shared issue/wait token", path=f"{path}.core_streams[{stream_index}].records")
            sequence = sequence_by_stream.setdefault(stream_index, [])
            sequence.append(action_id)
        for stream_index, action_ids in sequence_by_stream.items():
            by_first_record = sorted(action_ids, key=lambda action_id: records_by_action[action_id][1][0])
            expected = sorted(action_ids, key=lambda action_id: actions[action_id].core_order_index)
            if by_first_record != expected:
                raise SchemaError("stream action order violates GlobalAction core order", path=f"{path}.core_streams[{stream_index}].records")
        actual_plan_runtime_symbols = {
            symbol.id: symbol
            for symbol in self.runtime_symbols
            if symbol.source_ref in claimed_plan_barrier_ids
        }
        if actual_plan_runtime_symbols != expected_plan_runtime_symbols:
            raise SchemaError(
                "PLAN barrier runtime symbols must exactly equal canonical participant cores and directed pair events",
                path=f"{path}.runtime_symbols",
            )

        abi_by_schedule_binding = {
            (abi.schedule_id, abi.binding_id): abi for abi in self.buffer_abi
        }
        for stream_index, stream in enumerate(self.core_streams):
            for relocation_index, relocation in enumerate(stream.address_relocations):
                record = stream.records[relocation.record_index]
                if record.opcode in _LIFECYCLE_OPCODES:
                    continue
                action = actions[record.source_global_action_id]
                if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                    state_use = (
                        action.state_uses[0]
                        if len(action.state_uses) == 1
                        else None
                    )
                    if (
                        action.dma is None
                        or state_use is None
                        or program_symbols[relocation.symbol_ref].source_ref
                        != state_use.hbm_binding_ref
                        or relocation.addend != action.dma.state_offset_bytes
                    ):
                        raise SchemaError(
                            "HBM relocation does not exactly select its GlobalAction state endpoint",
                            path=f"{path}.core_streams[{stream_index}].address_relocations[{relocation_index}]",
                        )
                    continue
                role, operand_index = _address_operand_role(
                    record.opcode, relocation.operand_id, path
                )
                _uses, abis, addends, _lengths = _expected_operand_views(
                    action,
                    role,
                    operand_index,
                    abi_by_schedule_binding,
                    f"{path}.core_streams[{stream_index}].address_relocations[{relocation_index}]",
                )
                expected_addend = (
                    0
                    if relocation.symbol_kind is ProgramSymbolKind.SRAM_LABEL
                    else addends[0]
                    if relocation.symbol_kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                    else abis[0].region_offset_bytes + addends[0]
                )
                if relocation.addend != expected_addend:
                    raise SchemaError(
                        "address relocation addend does not exactly select its ActionBufferUse view",
                        path=f"{path}.core_streams[{stream_index}].address_relocations[{relocation_index}].addend",
                    )


@dataclass(frozen=True, slots=True)
class RegionManifest:
    schema_version: str
    producer_pass: str
    id: str
    region_id: str
    fusion_plan_id: str
    target_dies: tuple[int, ...]
    fragment: CommandFragment

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "RegionManifest":
        return cls(
            schema_version=REGION_MANIFEST_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("region_manifest", semantic_key, schema_version=REGION_MANIFEST_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in ("region_id", "fusion_plan_id", "target_dies", "fragment")}

    def validate(self, path: str = "region_manifest") -> None:
        if self.schema_version != REGION_MANIFEST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for field_name in ("producer_pass", "region_id", "fusion_plan_id"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        self.fragment.validate(f"{path}.fragment")
        if self.fragment.kind is not FragmentKind.ISA_REGION:
            raise SchemaError("region fragment must have ISA_REGION kind", path=f"{path}.fragment.kind")
        derived_dies = tuple(sorted({stream.logical_core.die_id for stream in self.fragment.core_streams}))
        if self.target_dies != derived_dies:
            raise SchemaError("target dies must exactly derive from fragment streams", path=f"{path}.target_dies")
        expected_id = stable_artifact_id("region_manifest", self._semantic_key(), schema_version=REGION_MANIFEST_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(self, dag: GlobalActionDAG, path: str = "region_manifest") -> None:
        self.validate(path)
        self.fragment.validate_against(dag, f"{path}.fragment")
        actions = {action.id: action for action in dag.actions}
        for action_id in self.fragment.claimed_action_ids:
            action = actions[action_id]
            if action.region_id != self.region_id or action.lowering is not RegionLowering.ISA_REGION:
                raise SchemaError("all claimed actions must belong to this one ISA region", path=f"{path}.region_id")
            if not isinstance(action.origin_ref, FusedNodeOrigin) or action.origin_ref.plan_id != self.fusion_plan_id:
                raise SchemaError("all claimed actions must belong to this one fusion plan", path=f"{path}.fusion_plan_id")


LinkedFragment = CommandFragment | RegionManifest


class ManifestInputKind(str, Enum):
    S3_LITE_MOE = "s3_lite_moe"
    S2_LITE_ROOTED_AR = "s2_lite_rooted_ar"
    TRAIN_LOWERED_PROGRAM = "train_lowered_program"
    IR1 = "ir1"
    FUSION_PLAN = "fusion_plan"
    STANDALONE_PLAN = "standalone_plan"
    IR2_PROJECTION = "ir2_projection"
    SCHEDULE_SET = "schedule_set"
    GLOBAL_ACTION_DAG = "global_action_dag"
    COMMAND_FRAGMENT = "command_fragment"
    REGION_MANIFEST = "region_manifest"


class EmptyCoreAckPolicy(str, Enum):
    EXCLUDE_EMPTY = "exclude_empty"
    INCLUDE_EMPTY = "include_empty"


class ProgramFailurePolicy(str, Enum):
    ABORT_ALL = "abort_all"


@dataclass(frozen=True, slots=True)
class ManifestInputDigest:
    kind: ManifestInputKind
    artifact_id: str
    schema_version: str
    digest: str

    def validate(self, path: str) -> None:
        for field_name in ("artifact_id", "schema_version"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if (
            len(self.digest) != 64
            or self.digest.lower() != self.digest
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise SchemaError("must be a lowercase SHA-256 hex digest", path=f"{path}.digest")


@dataclass(frozen=True, slots=True)
class CoreRuntimeBinding:
    logical_core: LogicalCoreRef
    core_spec_ref: str
    runtime_core_id: int
    sram_profile_ref: str

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        validate_nonempty(self.core_spec_ref, f"{path}.core_spec_ref")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("must fit ProgramArtifact uint16 core id", path=f"{path}.runtime_core_id")
        validate_nonempty(self.sram_profile_ref, f"{path}.sram_profile_ref")


@dataclass(frozen=True, slots=True)
class LinkedRecordRef:
    fragment_id: str
    fragment_record_index: int
    source_global_action_id: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.fragment_id, f"{path}.fragment_id")
        validate_uint64(self.fragment_record_index, f"{path}.fragment_record_index")
        validate_nonempty(self.source_global_action_id, f"{path}.source_global_action_id")


@dataclass(frozen=True, slots=True)
class LinkedCoreStream:
    logical_core: LogicalCoreRef
    runtime_core_id: int
    records: tuple[LinkedRecordRef, ...]

    def validate(self, path: str) -> None:
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        if self.runtime_core_id > 0xFFFF:
            raise SchemaError("must fit ProgramArtifact uint16 core id", path=f"{path}.runtime_core_id")
        for index, record in enumerate(self.records):
            record.validate(f"{path}.records[{index}]")


@dataclass(frozen=True, slots=True)
class EventCredit:
    symbol_ref: str
    count: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.count")


def _validate_canonical_strings(values: tuple[str, ...], path: str) -> None:
    if values != tuple(sorted(set(values))):
        raise SchemaError("must contain unique strings in canonical order", path=path)
    for index, value in enumerate(values):
        validate_nonempty(value, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class FragmentInterface:
    fragment_id: str
    runtime_imports: tuple[str, ...]
    runtime_exports: tuple[str, ...]
    program_imports: tuple[str, ...]
    program_exports: tuple[str, ...]
    entry_events: tuple[EventCredit, ...]
    exit_events: tuple[EventCredit, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.fragment_id, f"{path}.fragment_id")
        for field_name in (
            "runtime_imports",
            "runtime_exports",
            "program_imports",
            "program_exports",
        ):
            _validate_canonical_strings(getattr(self, field_name), f"{path}.{field_name}")
        if set(self.runtime_imports).intersection(self.runtime_exports):
            raise SchemaError("runtime import/export sets must be disjoint", path=path)
        if set(self.program_imports).intersection(self.program_exports):
            raise SchemaError("program import/export sets must be disjoint", path=path)
        for field_name in ("entry_events", "exit_events"):
            credits = getattr(self, field_name)
            if tuple(credit.symbol_ref for credit in credits) != tuple(
                sorted({credit.symbol_ref for credit in credits})
            ):
                raise SchemaError("event credits must have unique canonical symbols", path=f"{path}.{field_name}")
            for index, credit in enumerate(credits):
                credit.validate(f"{path}.{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class RuntimeSymbolDefinition:
    symbol: RuntimeSymbol
    logical_cores: tuple[LogicalCoreRef, ...]
    source_action_id: str | None
    destination_action_id: str | None

    def validate(self, path: str) -> None:
        self.symbol.validate(f"{path}.symbol")
        canonical_cores = tuple(
            sorted(set(self.logical_cores), key=lambda core: (core.die_id, core.local_core_id))
        )
        if self.logical_cores != canonical_cores:
            raise SchemaError("logical cores must be unique and canonical", path=f"{path}.logical_cores")
        for index, core in enumerate(self.logical_cores):
            core.validate(f"{path}.logical_cores[{index}]")
        for field_name in ("source_action_id", "destination_action_id"):
            value = getattr(self, field_name)
            if value is not None:
                validate_nonempty(value, f"{path}.{field_name}")

        kind = self.symbol.kind
        if kind in (RuntimeSymbolKind.RUNTIME_CORE, RuntimeSymbolKind.START_TAG):
            if len(self.logical_cores) != 1 or self.source_action_id is not None or self.destination_action_id is not None:
                raise SchemaError("runtime-core/start-tag definition requires one core and no action endpoints", path=path)
        elif kind is RuntimeSymbolKind.GROUP:
            if not self.logical_cores or len({core.die_id for core in self.logical_cores}) != 1:
                raise SchemaError("group definition requires non-empty same-die members", path=f"{path}.logical_cores")
            if self.source_action_id is not None or self.destination_action_id is not None:
                raise SchemaError("group definition cannot carry action endpoints", path=path)
        elif kind is RuntimeSymbolKind.EVENT_TAG:
            if len(self.logical_cores) != 2 or self.source_action_id is None or self.destination_action_id is None:
                raise SchemaError("event definition requires source/destination cores and actions", path=path)
        elif kind is RuntimeSymbolKind.DTE_FSM:
            if len(self.logical_cores) != 2 or self.source_action_id is None or self.destination_action_id is None:
                raise SchemaError("DTE FSM definition requires endpoint cores and actions", path=path)
        elif kind is RuntimeSymbolKind.DTE_TOKEN:
            if len(self.logical_cores) != 1 or self.source_action_id is None:
                raise SchemaError("DTE token definition requires one owning core and issuing action", path=path)


@dataclass(frozen=True, slots=True)
class ProgramSymbolDefinition:
    symbol: ProgramSymbol
    name: str
    value: int
    size_bytes: int
    logical_cores: tuple[LogicalCoreRef, ...]

    def validate(self, path: str) -> None:
        self.symbol.validate(f"{path}.symbol")
        validate_nonempty(self.name, f"{path}.name")
        if "\x00" in self.name:
            raise SchemaError("program symbol name cannot contain NUL", path=f"{path}.name")
        try:
            encoded_name = self.name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SchemaError(
                "program symbol name must be valid UTF-8 without surrogate code points",
                path=f"{path}.name",
            ) from error
        if len(encoded_name) > 255:
            raise SchemaError(
                "program symbol name must fit the 255-byte ProgramArtifact field",
                path=f"{path}.name",
            )
        validate_uint64(self.value, f"{path}.value")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        canonical_cores = tuple(
            sorted(set(self.logical_cores), key=lambda core: (core.die_id, core.local_core_id))
        )
        if not self.logical_cores or self.logical_cores != canonical_cores:
            raise SchemaError("logical cores must be non-empty, unique and canonical", path=f"{path}.logical_cores")
        for index, core in enumerate(self.logical_cores):
            core.validate(f"{path}.logical_cores[{index}]")
        if self.symbol.kind is ProgramSymbolKind.SRAM_REGION:
            if len(encoded_name) > 64 or self.size_bytes == 0:
                raise SchemaError("SRAM region requires a <=64-byte name and non-zero size", path=path)
        elif self.symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            if self.value != 0 or self.size_bytes != 0:
                raise SchemaError(
                    "SRAM label definitions carry no physical address or span",
                    path=path,
                )
        elif self.size_bytes == 0:
            raise SchemaError("absolute address symbol requires a non-zero span", path=f"{path}.size_bytes")
        if self.size_bytes and self.value > ((1 << 64) - 1) - (self.size_bytes - 1):
            raise SchemaError("symbol physical span overflows uint64", path=path)


@dataclass(frozen=True, slots=True)
class AddressOperandBinding:
    fragment_id: str
    logical_core: LogicalCoreRef
    fragment_record_index: int
    operand_id: SemanticOperandId
    buffer_abi_ids: tuple[str, ...]
    tensor_slices: tuple[TensorSlice, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.fragment_id, f"{path}.fragment_id")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(self.fragment_record_index, f"{path}.fragment_record_index")
        if not self.buffer_abi_ids:
            raise SchemaError("address operand requires at least one BufferABI", path=f"{path}.buffer_abi_ids")
        if len(set(self.buffer_abi_ids)) != len(self.buffer_abi_ids):
            raise SchemaError("address operand cannot repeat a BufferABI", path=f"{path}.buffer_abi_ids")
        if len(self.tensor_slices) != len(self.buffer_abi_ids):
            raise SchemaError(
                "tensor slices must bijectively match BufferABI ids",
                path=f"{path}.tensor_slices",
            )
        for index, abi_id in enumerate(self.buffer_abi_ids):
            validate_nonempty(abi_id, f"{path}.buffer_abi_ids[{index}]")
            tensor_slice = self.tensor_slices[index]
            if type(tensor_slice) is not TensorSlice:
                raise SchemaError(
                    "must be a TensorSlice",
                    path=f"{path}.tensor_slices[{index}]",
                )
            tensor_slice.validate(f"{path}.tensor_slices[{index}]")


@dataclass(frozen=True, slots=True)
class StateOperandBinding:
    """Exact HBM witness for one linked LSU HBM_ADDRESS operand."""

    fragment_id: str
    logical_core: LogicalCoreRef
    fragment_record_index: int
    operand_id: SemanticOperandId
    state_abi_id: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.fragment_id, f"{path}.fragment_id")
        self.logical_core.validate(f"{path}.logical_core")
        validate_uint64(
            self.fragment_record_index,
            f"{path}.fragment_record_index",
        )
        if self.operand_id is not SemanticOperandId.HBM_ADDRESS:
            raise SchemaError(
                "state operand binding only supports HBM_ADDRESS",
                path=f"{path}.operand_id",
            )
        validate_nonempty(self.state_abi_id, f"{path}.state_abi_id")


def _ordered_operand_uses(
    action: GlobalAction,
    role: BufferUseRole,
    operand_index: int,
    path: str,
):
    matching_uses = [
        use
        for use in action.buffer_uses
        if use.role is role
        and (operand_index < 0 or use.operand_index == operand_index)
    ]
    if role is BufferUseRole.REDUCE_INPUT:
        if action.reduction is None:
            raise SchemaError(
                "REDUCE_INPUT closure requires a reduction contract", path=path
            )
        uses_by_rank = {
            use.contribution_rank: use
            for use in matching_uses
            if use.contribution_rank is not None
        }
        if (
            len(uses_by_rank) != len(matching_uses)
            or tuple(sorted(uses_by_rank))
            != tuple(sorted(action.reduction.input_ranks))
        ):
            raise SchemaError(
                "REDUCE_INPUT uses must bijectively cover reduction input_ranks",
                path=path,
            )
        return tuple(uses_by_rank[rank] for rank in action.reduction.input_ranks)
    if len(matching_uses) != 1:
        raise SchemaError(
            "non-reduction address operand requires exactly one scheduled BufferABI",
            path=path,
        )
    return tuple(matching_uses)


def _tensor_slice_size_bytes(tensor_slice: TensorSlice, dtype: DType, path: str) -> int:
    element_bytes = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}.get(dtype)
    if element_bytes is None:
        raise SchemaError("unsupported dense-view dtype", path=f"{path}.dtype")
    elements = 1
    for extent in tensor_slice.shape:
        elements *= extent
    return elements * element_bytes


def _expected_operand_views(
    action: GlobalAction,
    role: BufferUseRole,
    operand_index: int,
    abi_by_schedule_binding: dict[tuple[str, str], BufferABI],
    path: str,
):
    uses = _ordered_operand_uses(action, role, operand_index, path)
    try:
        expected = tuple(
            abi_by_schedule_binding[(action.source.schedule_id, use.binding_id)]
            for use in uses
        )
    except KeyError as error:
        raise SchemaError(
            "address operand use lacks its exact BufferABI",
            path=path,
        ) from error
    addends = tuple(
        dense_row_major_view_byte_addend(
            abi.tensor_slice,
            use.tensor_slice,
            abi.dtype,
            path=path,
        )
        for use, abi in zip(uses, expected)
    )
    lengths = tuple(
        _tensor_slice_size_bytes(use.tensor_slice, abi.dtype, path)
        for use, abi in zip(uses, expected)
    )
    return uses, expected, addends, lengths


def _address_operand_role(
    opcode: RecordOpcode,
    operand_id: SemanticOperandId,
    path: str,
) -> tuple[BufferUseRole, int]:
    if opcode is RecordOpcode.SRAM_BIND:
        if operand_id is SemanticOperandId.SRAM_BIND_OUTPUT:
            return BufferUseRole.COMP_OUTPUT, 0
        if operand_id in _SRAM_BIND_INPUT_OPERAND_IDS:
            return (
                BufferUseRole.COMP_INPUT,
                _SRAM_BIND_INPUT_OPERAND_IDS.index(operand_id),
            )
    mapping = {
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_AUX_ADDRESS): (BufferUseRole.COMP_INPUT, 2),
        (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
        (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
        (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
        (RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.SEND_SOURCE, 0),
        (RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.RECV_DESTINATION, 0),
        (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.REDUCE_INPUT, -1),
        (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.REDUCE_OUTPUT, 0),
        (RecordOpcode.DTE_ISSUE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.LOCAL_COPY_SOURCE, 0),
        (RecordOpcode.DTE_ISSUE, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.LOCAL_COPY_DESTINATION, 0),
        (RecordOpcode.LSU_LOAD, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.DMA_DESTINATION, 0),
        (RecordOpcode.LSU_STORE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.DMA_SOURCE, 0),
    }
    result = mapping.get((opcode, operand_id))
    if result is None:
        raise SchemaError(
            "address operand has no frozen GlobalAction buffer role",
            path=path,
        )
    return result


def _validate_address_operand_closure(
    closure: AddressOperandBinding,
    action: GlobalAction,
    record: RelocatableRecord,
    role: BufferUseRole,
    operand_index: int,
    abi_by_schedule_binding: dict[tuple[str, str], BufferABI],
    abi_by_id: dict[str, BufferABI],
    path: str,
) -> tuple[tuple[BufferABI, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the exact semantic ABI order after proving operand closure."""

    matching_uses, expected, addends, lengths = _expected_operand_views(
        action, role, operand_index, abi_by_schedule_binding, path
    )
    if closure.buffer_abi_ids != tuple(abi.id for abi in expected):
        raise SchemaError(
            "address operand BufferABI closure does not match GlobalAction operand roles",
            path=path,
        )
    if closure.tensor_slices != tuple(use.tensor_slice for use in matching_uses):
        raise SchemaError(
            "address operand tensor views do not exactly match GlobalAction uses",
            path=path,
        )
    if role is BufferUseRole.REDUCE_INPUT:
        first = expected[0]
        if any(
            (abi.schedule_id, abi.logical_core, abi.region_ref, abi.size_bytes)
            != (first.schedule_id, first.logical_core, first.region_ref, first.size_bytes)
            for abi in expected
        ) or any(
            right.region_offset_bytes + right_addend
            != left.region_offset_bytes + left_addend + left_length
            for left, right, left_addend, right_addend, left_length in zip(
                expected, expected[1:], addends, addends[1:], lengths
            )
        ):
            raise SchemaError(
                "REDUCE_INPUT BufferABIs must form one contiguous rank-major span",
                path=path,
            )
        operands = {operand.name: operand for operand in record.operands}
        if (
            operands["input_count"].literal_value != len(expected)
            or operands["input_stride_bytes"].literal_value != first.size_bytes
        ):
            raise SchemaError(
                "LOCAL_REDUCE input count/stride must match the rank-major BufferABI span",
                path=path,
            )
    if any(abi.id not in abi_by_id for abi in expected):
        raise SchemaError("operand closure references an unknown BufferABI", path=path)
    return expected, addends, lengths


def _validate_local_reduce_absolute_alignment(
    absolute_starts: tuple[int, ...], path: str, *, alignment: int = 2
) -> None:
    if not absolute_starts or any(start % alignment for start in absolute_starts):
        raise SchemaError(
            "LOCAL_REDUCE absolute closure starts must be dtype aligned",
            path=path,
        )


def _validate_runtime_binding_ref(
    definition: RuntimeSymbolDefinition,
    action: GlobalAction,
    field: RuntimeOperandField,
    path: str,
) -> None:
    binding = action.runtime_binding
    if binding is None:
        raise SchemaError(
            "runtime relocation requires the source GlobalAction runtime_binding",
            path=path,
        )
    if field in (RuntimeOperandField.DTE_FSM, RuntimeOperandField.PEER_CORE):
        expected = binding.channel_symbol
    elif field is RuntimeOperandField.DTE_TOKEN:
        expected = binding.token_symbol
    elif field in (
        RuntimeOperandField.EVENT_TAG,
        RuntimeOperandField.SOURCE_CORE,
        RuntimeOperandField.DESTINATION_CORE,
    ):
        expected = binding.event_symbol
    else:
        return
    if expected is None or definition.symbol.source_ref != expected:
        raise SchemaError(
            "runtime symbol source_ref must exactly preserve its schedule logical binding",
            path=path,
        )


def _validate_fused_recv_wait_token_closure(
    actions: dict[str, GlobalAction],
    fragments: tuple[CommandFragment, ...],
    runtime_definitions: dict[str, RuntimeSymbolDefinition],
    path: str,
) -> None:
    recv_by_wait, wait_by_recv = _fused_recv_wait_pairs(actions, path)
    token_records: dict[tuple[str, str], dict[SemanticTaskKind, str]] = {}
    for fragment in fragments:
        for stream in fragment.core_streams:
            for relocation in stream.runtime_relocations:
                if relocation.field is not RuntimeOperandField.DTE_TOKEN:
                    continue
                record = stream.records[relocation.record_index]
                action = actions[record.source_global_action_id]
                recv = (
                    action
                    if action.task_kind is SemanticTaskKind.RECV
                    and action.id in wait_by_recv
                    else recv_by_wait.get(action.id)
                )
                if recv is None:
                    continue
                wait = wait_by_recv[recv.id]
                definition = runtime_definitions.get(relocation.symbol_ref)
                if definition is None:
                    raise SchemaError(
                        "fused RECV/WAIT token relocation has no global definition",
                        path=path,
                    )
                _validate_runtime_binding_ref(definition, action, relocation.field, path)
                if (
                    action.task_kind
                    not in (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
                    or stream.logical_core != recv.logical_core
                    or definition.logical_cores != (recv.logical_core,)
                    or definition.source_action_id != recv.id
                    or definition.destination_action_id != wait.id
                ):
                    raise SchemaError(
                        "fused RECV/WAIT DTE token definition must exactly bind its owner, consumer and core",
                        path=path,
                    )
                roles = token_records.setdefault((recv.id, wait.id), {})
                previous = roles.setdefault(action.task_kind, definition.symbol.id)
                if previous != definition.symbol.id:
                    raise SchemaError(
                        "one fused action cannot relocate multiple DTE tokens",
                        path=path,
                    )

    expected_pairs = {(recv_id, wait.id) for recv_id, wait in wait_by_recv.items()}
    if set(token_records) != expected_pairs or any(
        set(roles) != {SemanticTaskKind.RECV, SemanticTaskKind.WAIT}
        or len(set(roles.values())) != 1
        for roles in token_records.values()
    ):
        raise SchemaError(
            "every waited fused RECV/WAIT pair must share one exact DTE token relocation",
            path=path,
        )


def _validate_local_reduce_containment_witness(
    fragment_symbols: tuple[ProgramSymbol, ...],
    program_definitions: dict[str, ProgramSymbolDefinition],
    logical_core: LogicalCoreRef,
    region_name: str,
    region_base: int,
    region_size: int,
    closure_start: int,
    closure_end: int,
    path: str,
) -> str:
    witnesses = [
        program_definitions[symbol.id]
        for symbol in fragment_symbols
        if symbol.kind is ProgramSymbolKind.SRAM_REGION
        and symbol.id in program_definitions
        and logical_core in program_definitions[symbol.id].logical_cores
        and (
            program_definitions[symbol.id].name,
            program_definitions[symbol.id].value,
            program_definitions[symbol.id].size_bytes,
        )
        == (region_name, region_base, region_size)
    ]
    if len(witnesses) != 1 or not (
        witnesses[0].value <= closure_start
        and closure_end <= witnesses[0].value + witnesses[0].size_bytes
    ):
        raise SchemaError(
            "LOCAL_REDUCE absolute closure requires one exact named SRAM region containment witness",
            path=path,
        )
    return witnesses[0].symbol.id


@dataclass(frozen=True, slots=True)
class LogicalCoreGroup:
    symbol_ref: str
    members: tuple[LogicalCoreRef, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.symbol_ref, f"{path}.symbol_ref")
        canonical = tuple(sorted(set(self.members), key=lambda core: (core.die_id, core.local_core_id)))
        if not self.members or self.members != canonical:
            raise SchemaError("members must be non-empty, unique and canonical", path=f"{path}.members")
        if len({core.die_id for core in self.members}) != 1:
            raise SchemaError("ProgramArtifact core groups cannot span dies", path=f"{path}.members")
        for index, member in enumerate(self.members):
            member.validate(f"{path}.members[{index}]")


@dataclass(frozen=True, slots=True)
class LogicalStartEvent:
    target_core: LogicalCoreRef
    tag_symbol_ref: str
    count: int

    def validate(self, path: str) -> None:
        self.target_core.validate(f"{path}.target_core")
        validate_nonempty(self.tag_symbol_ref, f"{path}.tag_symbol_ref")
        validate_uint64(self.count, f"{path}.count")
        if self.count == 0 or self.count > 0xFF:
            raise SchemaError("start count must fit the non-zero uint8 helper wire", path=f"{path}.count")


def _validate_canonical_cores(values: tuple[LogicalCoreRef, ...], path: str) -> None:
    canonical = tuple(sorted(set(values), key=lambda core: (core.die_id, core.local_core_id)))
    if values != canonical:
        raise SchemaError("cores must be unique and canonical", path=path)
    for index, core in enumerate(values):
        core.validate(f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ProgramControlEnvelope:
    active_cores: tuple[LogicalCoreRef, ...]
    start_events: tuple[LogicalStartEvent, ...]
    terminal_cores: tuple[LogicalCoreRef, ...]
    expected_ack_cores: tuple[LogicalCoreRef, ...]
    expected_done_cores: tuple[LogicalCoreRef, ...]
    empty_core_ack_policy: EmptyCoreAckPolicy
    failure_policy: ProgramFailurePolicy

    def validate(self, path: str) -> None:
        for field_name in (
            "active_cores",
            "terminal_cores",
            "expected_ack_cores",
            "expected_done_cores",
        ):
            _validate_canonical_cores(getattr(self, field_name), f"{path}.{field_name}")
        if not self.active_cores:
            raise SchemaError("must contain at least one active core", path=f"{path}.active_cores")
        if not self.terminal_cores:
            raise SchemaError("DATAFLOW requires at least one terminal core", path=f"{path}.terminal_cores")
        active = set(self.active_cores)
        for field_name in ("terminal_cores", "expected_ack_cores", "expected_done_cores"):
            if not set(getattr(self, field_name)).issubset(active):
                raise SchemaError("core set must be a subset of active cores", path=f"{path}.{field_name}")
        if self.expected_done_cores != self.terminal_cores:
            raise SchemaError("expected DONE cores must exactly equal terminal cores", path=f"{path}.expected_done_cores")
        keys: list[tuple[LogicalCoreRef, str]] = []
        for index, event in enumerate(self.start_events):
            event.validate(f"{path}.start_events[{index}]")
            if event.target_core not in active:
                raise SchemaError("start target lies outside active cores", path=f"{path}.start_events[{index}].target_core")
            keys.append((event.target_core, event.tag_symbol_ref))
        canonical_keys = sorted(keys, key=lambda item: (item[0].die_id, item[0].local_core_id, item[1]))
        if keys != canonical_keys or len(set(keys)) != len(keys):
            raise SchemaError("start events must be unique and canonical by target/tag", path=f"{path}.start_events")
        if len({event.tag_symbol_ref for event in self.start_events}) > 0x10000:
            raise SchemaError("start tag namespace cannot fit the helper uint16 wire", path=f"{path}.start_events")
        if self.failure_policy is not ProgramFailurePolicy.ABORT_ALL:
            raise SchemaError("ProgramArtifact v1 only supports ABORT_ALL", path=f"{path}.failure_policy")


@dataclass(frozen=True, slots=True)
class LinkedProgramManifest:
    """Exact symbolic link product consumed by the C++ ID/address finalizer."""

    schema_version: str
    producer_pass: str
    id: str
    capabilities: int
    source_ir1_id: str
    source_projection_id: str
    source_schedule_set_id: str
    source_global_dag_id: str
    input_digests: tuple[ManifestInputDigest, ...]
    fragments: tuple[LinkedFragment, ...]
    fragment_interfaces: tuple[FragmentInterface, ...]
    core_bindings: tuple[CoreRuntimeBinding, ...]
    core_streams: tuple[LinkedCoreStream, ...]
    runtime_symbol_definitions: tuple[RuntimeSymbolDefinition, ...]
    program_symbol_definitions: tuple[ProgramSymbolDefinition, ...]
    address_operand_bindings: tuple[AddressOperandBinding, ...]
    state_operand_bindings: tuple[StateOperandBinding, ...]
    core_groups: tuple[LogicalCoreGroup, ...]
    envelope: ProgramControlEnvelope

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "LinkedProgramManifest":
        semantic_key.setdefault("state_operand_bindings", ())
        return cls(
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("linked_program_manifest", semantic_key, schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "capabilities",
                "source_ir1_id",
                "source_projection_id",
                "source_schedule_set_id",
                "source_global_dag_id",
                "input_digests",
                "fragments",
                "fragment_interfaces",
                "core_bindings",
                "core_streams",
                "runtime_symbol_definitions",
                "program_symbol_definitions",
                "address_operand_bindings",
                "state_operand_bindings",
                "core_groups",
                "envelope",
            )
        }

    def validate(self, path: str = "linked_program_manifest") -> None:
        if self.schema_version != LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        for field_name in (
            "source_ir1_id",
            "source_projection_id",
            "source_schedule_set_id",
            "source_global_dag_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        validate_uint64(self.capabilities, f"{path}.capabilities")
        if self.capabilities != 0:
            raise SchemaError("MVP ProgramArtifact capabilities must be zero", path=f"{path}.capabilities")

        digest_keys: list[tuple[str, str]] = []
        for index, digest in enumerate(self.input_digests):
            digest.validate(f"{path}.input_digests[{index}]")
            digest_keys.append((digest.kind.value, digest.artifact_id))
        if digest_keys != sorted(set(digest_keys)):
            raise SchemaError("input digests must be unique and canonical", path=f"{path}.input_digests")

        train_inputs = tuple(
            digest
            for digest in self.input_digests
            if digest.kind is ManifestInputKind.TRAIN_LOWERED_PROGRAM
        )
        s3_lite_inputs = tuple(
            digest
            for digest in self.input_digests
            if digest.kind is ManifestInputKind.S3_LITE_MOE
        )
        rooted_ar_inputs = tuple(
            digest
            for digest in self.input_digests
            if digest.kind is ManifestInputKind.S2_LITE_ROOTED_AR
        )
        top_input_count = sum(
            bool(inputs)
            for inputs in (train_inputs, s3_lite_inputs, rooted_ar_inputs)
        )
        if (
            len(train_inputs) > 1
            or len(s3_lite_inputs) > 1
            or len(rooted_ar_inputs) > 1
            or top_input_count > 1
        ):
            raise SchemaError(
                "Train, S3-Lite and S2-Lite rooted-AR top-level inputs are exclusive and singular",
                path=f"{path}.input_digests",
            )
        train_lineage_ids = {
            kind: {
                digest.artifact_id
                for digest in self.input_digests
                if digest.kind is kind
            }
            for kind in (
                ManifestInputKind.IR1,
                ManifestInputKind.IR2_PROJECTION,
                ManifestInputKind.SCHEDULE_SET,
                ManifestInputKind.GLOBAL_ACTION_DAG,
            )
        }
        if train_inputs:
            lineage_cardinalities = {
                len(ids) for ids in train_lineage_ids.values()
            }
            if lineage_cardinalities == {0} or len(lineage_cardinalities) != 1:
                raise SchemaError(
                    "train input requires equal non-zero IR1/projection/schedule/global digest coverage",
                    path=f"{path}.input_digests",
                )
        if rooted_ar_inputs:
            rooted_lineage_count = {
                "wafer_frontend.s2_lite_rooted_ar_lowered_program/v1alpha1": 2,
                "wafer_frontend.s2_lite_dp4_tree_ar_lowered_program/v1alpha1": 4,
            }.get(rooted_ar_inputs[0].schema_version)
            if rooted_lineage_count is None:
                raise SchemaError(
                    "S2-Lite rooted-AR input has an unsupported top schema version",
                    path=f"{path}.input_digests",
                )
            if any(
                len(ids) != rooted_lineage_count
                for ids in train_lineage_ids.values()
            ):
                raise SchemaError(
                    "S2-Lite rooted-AR input has wrong exact replica lineage cardinality",
                    path=f"{path}.input_digests",
                )
        if s3_lite_inputs:
            if any(len(ids) != 1 for ids in train_lineage_ids.values()):
                raise SchemaError(
                    "S3-Lite input requires one exact IR1/projection/schedule/global lineage",
                    path=f"{path}.input_digests",
                )
            if (
                train_lineage_ids[ManifestInputKind.IR1] != {self.source_ir1_id}
                or train_lineage_ids[ManifestInputKind.IR2_PROJECTION] != {self.source_projection_id}
                or train_lineage_ids[ManifestInputKind.SCHEDULE_SET] != {self.source_schedule_set_id}
                or train_lineage_ids[ManifestInputKind.GLOBAL_ACTION_DAG] != {self.source_global_dag_id}
            ):
                raise SchemaError(
                    "S3-Lite lineage digests must identify the manifest sources",
                    path=f"{path}.input_digests",
                )

        if not self.fragments:
            raise SchemaError("must contain lowering fragments", path=f"{path}.fragments")
        fragment_ids: list[str] = []
        leaf_fragments: dict[str, CommandFragment] = {}
        for index, linked in enumerate(self.fragments):
            linked_path = f"{path}.fragments[{index}]"
            linked.validate(linked_path)
            fragment_ids.append(linked.id)
            leaf = linked.fragment if isinstance(linked, RegionManifest) else linked
            if leaf.id in leaf_fragments:
                raise SchemaError("duplicate leaf CommandFragment", path=linked_path)
            if rooted_ar_inputs:
                if leaf.kind is FragmentKind.S2_LITE_ROOTED_AR:
                    if leaf.source_global_dag_id != self.source_global_dag_id:
                        raise SchemaError(
                            "rooted-AR overlay fragment must reference the top carrier",
                            path=f"{linked_path}.source_global_dag_id",
                        )
                elif leaf.source_global_dag_id not in train_lineage_ids[
                    ManifestInputKind.GLOBAL_ACTION_DAG
                ]:
                    raise SchemaError(
                        "rooted-AR local fragment global DAG lacks an exact input digest",
                        path=f"{linked_path}.source_global_dag_id",
                    )
            elif train_inputs:
                if leaf.source_global_dag_id not in train_lineage_ids[
                    ManifestInputKind.GLOBAL_ACTION_DAG
                ]:
                    raise SchemaError(
                        "train fragment global DAG lacks an exact input digest",
                        path=f"{linked_path}.source_global_dag_id",
                    )
            elif leaf.source_global_dag_id != self.source_global_dag_id:
                raise SchemaError("fragment references a different global DAG", path=f"{linked_path}.source_global_dag_id")
            leaf_fragments[leaf.id] = leaf
        if train_inputs and {
            leaf.source_global_dag_id for leaf in leaf_fragments.values()
        } != train_lineage_ids[ManifestInputKind.GLOBAL_ACTION_DAG]:
            raise SchemaError(
                "train fragments must witness every replica global DAG digest",
                path=f"{path}.fragments",
            )
        if rooted_ar_inputs:
            local_dag_ids = {
                leaf.source_global_dag_id
                for leaf in leaf_fragments.values()
                if leaf.kind is not FragmentKind.S2_LITE_ROOTED_AR
            }
            overlay_fragments = tuple(
                leaf
                for leaf in leaf_fragments.values()
                if leaf.kind is FragmentKind.S2_LITE_ROOTED_AR
            )
            if local_dag_ids != train_lineage_ids[
                ManifestInputKind.GLOBAL_ACTION_DAG
            ]:
                raise SchemaError(
                    "rooted-AR local fragments must witness every replica global DAG digest",
                    path=f"{path}.fragments",
                )
            if not overlay_fragments:
                raise SchemaError(
                    "rooted-AR input requires dedicated overlay fragments",
                    path=f"{path}.fragments",
                )
        if fragment_ids != sorted(set(fragment_ids)):
            raise SchemaError("linked fragments must have unique canonical artifact ids", path=f"{path}.fragments")

        interface_ids: list[str] = []
        interfaces: dict[str, FragmentInterface] = {}
        for index, interface in enumerate(self.fragment_interfaces):
            interface.validate(f"{path}.fragment_interfaces[{index}]")
            interface_ids.append(interface.fragment_id)
            interfaces[interface.fragment_id] = interface
        if interface_ids != sorted(set(interface_ids)) or set(interfaces) != set(leaf_fragments):
            raise SchemaError("interfaces must bijectively cover leaf fragments in canonical order", path=f"{path}.fragment_interfaces")

        binding_cores: list[LogicalCoreRef] = []
        runtime_core_ids: list[int] = []
        binding_by_core: dict[LogicalCoreRef, CoreRuntimeBinding] = {}
        for index, binding in enumerate(self.core_bindings):
            binding.validate(f"{path}.core_bindings[{index}]")
            binding_cores.append(binding.logical_core)
            runtime_core_ids.append(binding.runtime_core_id)
            binding_by_core[binding.logical_core] = binding
        canonical_cores = sorted(set(binding_cores), key=lambda core: (core.die_id, core.local_core_id))
        if binding_cores != canonical_cores or len(set(runtime_core_ids)) != len(runtime_core_ids):
            raise SchemaError("core bindings require canonical logical cores and unique runtime ids", path=f"{path}.core_bindings")

        stream_cores: list[LogicalCoreRef] = []
        for index, stream in enumerate(self.core_streams):
            stream.validate(f"{path}.core_streams[{index}]")
            stream_cores.append(stream.logical_core)
            binding = binding_by_core.get(stream.logical_core)
            if binding is None or binding.runtime_core_id != stream.runtime_core_id:
                raise SchemaError("stream does not exactly match one core binding", path=f"{path}.core_streams[{index}]")
        if stream_cores != binding_cores:
            raise SchemaError("core streams must bijectively follow core bindings, including empty streams", path=f"{path}.core_streams")
        expected_record_refs = {
            (fragment_id, stream.logical_core, record_index, record.source_global_action_id)
            for fragment_id, fragment in leaf_fragments.items()
            for stream in fragment.core_streams
            for record_index, record in enumerate(stream.records)
        }
        actual_record_refs: list[tuple[str, LogicalCoreRef, int, str]] = []
        for stream in self.core_streams:
            for record in stream.records:
                actual_record_refs.append(
                    (
                        record.fragment_id,
                        stream.logical_core,
                        record.fragment_record_index,
                        record.source_global_action_id,
                    )
                )
        if (
            len(set(actual_record_refs)) != len(actual_record_refs)
            or set(actual_record_refs) != expected_record_refs
        ):
            raise SchemaError(
                "linked core streams must cover every fragment record exactly once",
                path=f"{path}.core_streams",
            )

        runtime_definitions: dict[str, RuntimeSymbolDefinition] = {}
        for index, definition in enumerate(self.runtime_symbol_definitions):
            definition.validate(f"{path}.runtime_symbol_definitions[{index}]")
            if definition.symbol.id in runtime_definitions:
                raise SchemaError("duplicate runtime symbol definition", path=f"{path}.runtime_symbol_definitions[{index}].symbol.id")
            runtime_definitions[definition.symbol.id] = definition
        if tuple(runtime_definitions) != tuple(sorted(runtime_definitions)):
            raise SchemaError("runtime symbol definitions must be canonical by id", path=f"{path}.runtime_symbol_definitions")

        program_definitions: dict[str, ProgramSymbolDefinition] = {}
        program_names: set[str] = set()
        for index, definition in enumerate(self.program_symbol_definitions):
            definition.validate(f"{path}.program_symbol_definitions[{index}]")
            if definition.symbol.id in program_definitions:
                raise SchemaError("duplicate program symbol definition", path=f"{path}.program_symbol_definitions[{index}].symbol.id")
            if definition.name in program_names:
                raise SchemaError("ProgramArtifact symbol names must be globally unique", path=f"{path}.program_symbol_definitions[{index}].name")
            program_definitions[definition.symbol.id] = definition
            program_names.add(definition.name)
        if tuple(program_definitions) != tuple(sorted(program_definitions)):
            raise SchemaError("program symbol definitions must be canonical by id", path=f"{path}.program_symbol_definitions")

        runtime_exporters: dict[str, str] = {}
        program_exporters: dict[str, str] = {}
        for fragment_id, fragment in leaf_fragments.items():
            interface = interfaces[fragment_id]
            local_runtime = {symbol.id: symbol for symbol in fragment.runtime_symbols}
            local_program = {symbol.id: symbol for symbol in fragment.program_symbols}
            if set(interface.runtime_imports + interface.runtime_exports) != set(local_runtime):
                raise SchemaError("runtime import/export closure must equal local declarations", path=f"{path}.fragment_interfaces")
            if set(interface.program_imports + interface.program_exports) != set(local_program):
                raise SchemaError("program import/export closure must equal local declarations", path=f"{path}.fragment_interfaces")
            for symbol_id, symbol in local_runtime.items():
                definition = runtime_definitions.get(symbol_id)
                if definition is None or definition.symbol != symbol:
                    raise SchemaError("local runtime declaration has no identical global definition", path=f"{path}.runtime_symbol_definitions")
            for symbol_id, symbol in local_program.items():
                definition = program_definitions.get(symbol_id)
                if definition is None or definition.symbol != symbol:
                    raise SchemaError("local program declaration has no identical global definition", path=f"{path}.program_symbol_definitions")
            for symbol_id in interface.runtime_exports:
                if symbol_id in runtime_exporters:
                    raise SchemaError("runtime symbol must have exactly one exporting fragment", path=f"{path}.fragment_interfaces")
                runtime_exporters[symbol_id] = fragment_id
            for symbol_id in interface.program_exports:
                if symbol_id in program_exporters:
                    raise SchemaError("program symbol must have exactly one exporting fragment", path=f"{path}.fragment_interfaces")
                program_exporters[symbol_id] = fragment_id
        declared_runtime = {symbol.id for fragment in leaf_fragments.values() for symbol in fragment.runtime_symbols}
        declared_program = {symbol.id for fragment in leaf_fragments.values() for symbol in fragment.program_symbols}
        if set(runtime_exporters) != declared_runtime:
            raise SchemaError("every fragment runtime symbol needs exactly one exporter", path=f"{path}.fragment_interfaces")
        if set(program_exporters) != declared_program:
            raise SchemaError("every fragment program symbol needs exactly one exporter", path=f"{path}.fragment_interfaces")

        buffer_abi: dict[str, BufferABI] = {}
        schedule_binding_to_id: dict[tuple[str, str], str] = {}
        state_abi: dict[str, StateABI] = {}
        state_binding_to_id: dict[str, str] = {}
        for fragment in leaf_fragments.values():
            for abi in fragment.buffer_abi:
                previous = buffer_abi.get(abi.id)
                if previous is not None and previous != abi:
                    raise SchemaError("conflicting shared BufferABI definition", path=f"{path}.fragments")
                key = (abi.schedule_id, abi.binding_id)
                previous_id = schedule_binding_to_id.get(key)
                if previous_id is not None and previous_id != abi.id:
                    raise SchemaError("one schedule binding cannot have multiple BufferABI ids", path=f"{path}.fragments")
                buffer_abi[abi.id] = abi
                schedule_binding_to_id[key] = abi.id
            for abi in fragment.state_abi:
                previous = state_abi.get(abi.id)
                if previous is not None and previous != abi:
                    raise SchemaError(
                        "conflicting shared StateABI definition",
                        path=f"{path}.fragments",
                    )
                previous_id = state_binding_to_id.get(abi.hbm_binding_ref)
                if previous_id is not None and previous_id != abi.id:
                    raise SchemaError(
                        "one HBM binding cannot have multiple StateABI ids",
                        path=f"{path}.fragments",
                    )
                state_abi[abi.id] = abi
                state_binding_to_id[abi.hbm_binding_ref] = abi.id

        address_keys: list[tuple[int, int, str, int, int]] = []
        address_bindings: dict[tuple[str, LogicalCoreRef, int, SemanticOperandId], AddressOperandBinding] = {}
        for index, binding in enumerate(self.address_operand_bindings):
            binding.validate(f"{path}.address_operand_bindings[{index}]")
            key = (binding.fragment_id, binding.logical_core, binding.fragment_record_index, binding.operand_id)
            if key in address_bindings:
                raise SchemaError("duplicate address operand binding", path=f"{path}.address_operand_bindings[{index}]")
            if binding.operand_id is SemanticOperandId.HBM_ADDRESS:
                raise SchemaError("HBM_ADDRESS cannot use a BufferABI closure", path=f"{path}.address_operand_bindings[{index}].operand_id")
            if not set(binding.buffer_abi_ids).issubset(buffer_abi):
                raise SchemaError("address operand references an unknown BufferABI", path=f"{path}.address_operand_bindings[{index}].buffer_abi_ids")
            address_bindings[key] = binding
            address_keys.append((binding.logical_core.die_id, binding.logical_core.local_core_id, binding.fragment_id, binding.fragment_record_index, int(binding.operand_id)))
        if address_keys != sorted(address_keys):
            raise SchemaError("address operand bindings must be canonical by core/fragment/record/operand", path=f"{path}.address_operand_bindings")

        state_keys: list[tuple[int, int, str, int, int]] = []
        state_bindings: dict[tuple[str, LogicalCoreRef, int, SemanticOperandId], StateOperandBinding] = {}
        for index, binding in enumerate(self.state_operand_bindings):
            binding.validate(f"{path}.state_operand_bindings[{index}]")
            key = (binding.fragment_id, binding.logical_core, binding.fragment_record_index, binding.operand_id)
            if key in state_bindings:
                raise SchemaError("duplicate state operand binding", path=f"{path}.state_operand_bindings[{index}]")
            if binding.state_abi_id not in state_abi:
                raise SchemaError("state operand references an unknown StateABI", path=f"{path}.state_operand_bindings[{index}].state_abi_id")
            state_bindings[key] = binding
            state_keys.append((binding.logical_core.die_id, binding.logical_core.local_core_id, binding.fragment_id, binding.fragment_record_index, int(binding.operand_id)))
        if state_keys != sorted(state_keys):
            raise SchemaError("state operand bindings must be canonical by core/fragment/record/operand", path=f"{path}.state_operand_bindings")

        buffer_relocation_keys: set[tuple[str, LogicalCoreRef, int, SemanticOperandId]] = set()
        state_relocation_keys: set[tuple[str, LogicalCoreRef, int, SemanticOperandId]] = set()
        used_state_abi_ids: set[str] = set()
        for fragment_id, fragment in leaf_fragments.items():
            for stream in fragment.core_streams:
                for relocation in stream.address_relocations:
                    key = (fragment_id, stream.logical_core, relocation.record_index, relocation.operand_id)
                    definition = program_definitions.get(relocation.symbol_ref)
                    if definition is None:
                        raise SchemaError("address relocation lacks a program symbol definition", path=f"{path}.program_symbol_definitions")
                    if stream.logical_core not in definition.logical_cores:
                        raise SchemaError("program symbol definition omits the executing core", path=f"{path}.program_symbol_definitions")
                    if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                        state_relocation_keys.add(key)
                        if key in address_bindings:
                            raise SchemaError("HBM_ADDRESS cannot also use a BufferABI closure", path=f"{path}.address_operand_bindings")
                        closure = state_bindings.get(key)
                        if closure is None:
                            raise SchemaError("every HBM_ADDRESS relocation requires one StateABI closure", path=f"{path}.state_operand_bindings")
                        abi = state_abi[closure.state_abi_id]
                        if relocation.record_index >= len(stream.records):
                            raise SchemaError("state operand references a dangling record", path=f"{path}.state_operand_bindings")
                        record = stream.records[relocation.record_index]
                        if record.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                            raise SchemaError("HBM_ADDRESS is only legal on blocking LSU records", path=f"{path}.state_operand_bindings")
                        if (
                            relocation.symbol_kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                            or definition.symbol.kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                            or definition.symbol.source_ref != abi.hbm_binding_ref
                            or definition.value != abi.address
                            or definition.size_bytes != abi.size_bytes
                            or stream.logical_core.die_id != abi.die_id
                        ):
                            raise SchemaError("HBM relocation/definition must exactly preserve StateABI", path=f"{path}.state_operand_bindings")
                        size_bytes = record.operands[1].literal_value
                        assert size_bytes is not None
                        if (
                            relocation.addend > abi.size_bytes
                            or size_bytes > abi.size_bytes - relocation.addend
                        ):
                            raise SchemaError(
                                "HBM relocation byte range exceeds StateABI",
                                path=f"{path}.state_operand_bindings",
                            )
                        if (
                            record.opcode is RecordOpcode.LSU_LOAD
                            and abi.access is PersistentStateAccess.RESERVED
                        ) or (
                            record.opcode is RecordOpcode.LSU_STORE
                            and abi.access is not PersistentStateAccess.READ_WRITE
                        ):
                            raise SchemaError("blocking LSU direction is forbidden by StateABI access", path=f"{path}.state_operand_bindings")
                        used_state_abi_ids.add(abi.id)
                    else:
                        buffer_relocation_keys.add(key)
                        if key in state_bindings:
                            raise SchemaError("non-HBM address cannot use a StateABI closure", path=f"{path}.state_operand_bindings")
                        if key not in address_bindings:
                            raise SchemaError("every non-HBM address relocation requires one BufferABI closure", path=f"{path}.address_operand_bindings")
        if buffer_relocation_keys != set(address_bindings):
            raise SchemaError("address operand binding has no matching relocation", path=f"{path}.address_operand_bindings")
        if state_relocation_keys != set(state_bindings):
            raise SchemaError("state operand binding has no matching HBM relocation", path=f"{path}.state_operand_bindings")
        if used_state_abi_ids != set(state_abi):
            raise SchemaError("every StateABI requires a linked HBM operand witness", path=f"{path}.state_operand_bindings")

        group_symbols: list[str] = []
        for index, group in enumerate(self.core_groups):
            group.validate(f"{path}.core_groups[{index}]")
            group_symbols.append(group.symbol_ref)
            definition = runtime_definitions.get(group.symbol_ref)
            if definition is None or definition.symbol.kind is not RuntimeSymbolKind.GROUP or definition.logical_cores != group.members:
                raise SchemaError("core group must exactly match one GROUP symbol definition", path=f"{path}.core_groups[{index}]")
            if not set(group.members).issubset(binding_by_core):
                raise SchemaError("core group member lies outside active core bindings", path=f"{path}.core_groups[{index}].members")
        if group_symbols != sorted(set(group_symbols)):
            raise SchemaError("core groups must be unique and canonical by symbol", path=f"{path}.core_groups")

        derived_entries: dict[tuple[str, str], int] = {}
        derived_exits: dict[tuple[str, str], int] = {}
        for fragment_id, fragment in leaf_fragments.items():
            for stream in fragment.core_streams:
                for record in stream.records:
                    operands = {operand.name: operand for operand in record.operands}
                    if record.opcode is RecordOpcode.EVENT_SET:
                        symbol_ref = operands["tag"].symbol_ref
                        assert symbol_ref is not None
                        derived_exits[(fragment_id, symbol_ref)] = derived_exits.get((fragment_id, symbol_ref), 0) + 1
                    elif record.opcode is RecordOpcode.EVENT_WAIT:
                        symbol_ref = operands["tag"].symbol_ref
                        count = operands["count"].literal_value
                        assert symbol_ref is not None and type(count) is int
                        derived_entries[(fragment_id, symbol_ref)] = derived_entries.get((fragment_id, symbol_ref), 0) + count
        actual_entries = {
            (interface.fragment_id, credit.symbol_ref): credit.count
            for interface in self.fragment_interfaces for credit in interface.entry_events
        }
        actual_exits = {
            (interface.fragment_id, credit.symbol_ref): credit.count
            for interface in self.fragment_interfaces for credit in interface.exit_events
        }
        if actual_entries != derived_entries or actual_exits != derived_exits:
            raise SchemaError("fragment entry/exit event credits must exactly derive from records", path=f"{path}.fragment_interfaces")
        event_symbols = set(symbol for _fragment, symbol in derived_entries).union(
            symbol for _fragment, symbol in derived_exits
        )
        for symbol_ref in event_symbols:
            definition = runtime_definitions.get(symbol_ref)
            if definition is None or definition.symbol.kind is not RuntimeSymbolKind.EVENT_TAG:
                raise SchemaError("event credit references a non-EVENT_TAG symbol", path=f"{path}.fragment_interfaces")
            sets = sum(count for (_fragment, symbol), count in derived_exits.items() if symbol == symbol_ref)
            waits = sum(count for (_fragment, symbol), count in derived_entries.items() if symbol == symbol_ref)
            if sets != waits:
                raise SchemaError("EVENT SET/WAIT credits are imbalanced", path=f"{path}.fragment_interfaces")
            if sets != 1:
                raise SchemaError(
                    "MVP requires exactly one EVENT_SET and one EVENT_WAIT per logical event",
                    path=f"{path}.fragment_interfaces",
                )

        self.envelope.validate(f"{path}.envelope")
        if self.envelope.active_cores != tuple(binding_cores):
            raise SchemaError("active cores must exactly equal the program core bindings", path=f"{path}.envelope.active_cores")
        start_symbols = {event.tag_symbol_ref for event in self.envelope.start_events}
        for event in self.envelope.start_events:
            definition = runtime_definitions.get(event.tag_symbol_ref)
            if definition is None or definition.symbol.kind is not RuntimeSymbolKind.START_TAG or definition.logical_cores != (event.target_core,):
                raise SchemaError("start event must resolve to an exact START_TAG definition", path=f"{path}.envelope.start_events")
        allowed_top_only = start_symbols.union(group_symbols)
        top_only = set(runtime_definitions).difference(declared_runtime)
        if not start_symbols.issubset(top_only) or not top_only.issubset(allowed_top_only):
            raise SchemaError("unreferenced top-level runtime symbol definition", path=f"{path}.runtime_symbol_definitions")
        if set(program_definitions) != declared_program:
            raise SchemaError("unreferenced top-level program symbol definition", path=f"{path}.program_symbol_definitions")

        nonempty_cores = {stream.logical_core for stream in self.core_streams if stream.records}
        expected_ack = (
            self.envelope.active_cores
            if self.envelope.empty_core_ack_policy is EmptyCoreAckPolicy.INCLUDE_EMPTY
            else tuple(core for core in self.envelope.active_cores if core in nonempty_cores)
        )
        if self.envelope.expected_ack_cores != expected_ack:
            raise SchemaError("expected ACK cores do not close under empty-core policy", path=f"{path}.envelope.expected_ack_cores")
        if self.envelope.empty_core_ack_policy is EmptyCoreAckPolicy.EXCLUDE_EMPTY:
            empty = set(self.envelope.active_cores).difference(nonempty_cores)
            if empty:
                raise SchemaError("MVP EXCLUDE_EMPTY manifest rejects active empty streams", path=f"{path}.core_streams")

        expected_id = stable_artifact_id(
            "linked_program_manifest",
            self._semantic_key(),
            schema_version=LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")
    def validate_against(
        self,
        ir1: IR1,
        fusion_plans: tuple[FusionPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
        projection: IR2ProjectionResult,
        schedule_set: IntraDieScheduleSet,
        dag: GlobalActionDAG,
        fragments: tuple[LinkedFragment, ...],
        path: str = "linked_program_manifest",
    ) -> None:
        """Prove the symbolic link product is an exact quotient of all inputs."""

        self.validate(path)
        ir1.validate("ir1")
        if tuple(plan.id for plan in fusion_plans) != projection.fusion_plan_ids:
            raise SchemaError("fusion plan tuple must exactly preserve projection order", path=f"{path}.input_digests")
        if tuple(plan.id for plan in standalone_plans) != projection.standalone_collective_plan_ids:
            raise SchemaError("standalone plan tuple must exactly preserve projection order", path=f"{path}.input_digests")
        for index, plan in enumerate(fusion_plans):
            plan.validate_against(ir1, f"fusion_plans[{index}]")
        for index, plan in enumerate(standalone_plans):
            plan.validate_against(ir1, f"standalone_plans[{index}]")
        projection.validate_against(ir1, fusion_plans, standalone_plans, "ir2_projection_result")
        schedule_set.validate_against(projection, ir1, "intra_die_schedule_set")
        dag.validate_against(ir1, projection, schedule_set, "global_action_dag")

        used_hbm_bindings = {
            use.hbm_binding_ref
            for action in dag.actions
            for use in action.state_uses
        }
        actual_state_by_binding: dict[str, StateABI] = {}
        for linked in fragments:
            leaf = linked.fragment if isinstance(linked, RegionManifest) else linked
            for abi in leaf.state_abi:
                previous = actual_state_by_binding.get(abi.hbm_binding_ref)
                if previous is not None and previous != abi:
                    raise SchemaError(
                        "lowering fragments contain conflicting StateABI definitions",
                        path=f"{path}.fragments",
                    )
                actual_state_by_binding[abi.hbm_binding_ref] = abi
        manifest = ir1.persistent_state_manifest
        if used_hbm_bindings:
            if manifest is None:
                raise SchemaError(
                    "state actions require the IR1 persistent-state manifest",
                    path=f"{path}.fragments",
                )
            declarations = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }
            bindings = {binding.id: binding for binding in manifest.bindings}
            spaces = {space.die_id: space for space in manifest.address_spaces}
            expected_state_by_binding: dict[str, StateABI] = {}
            for binding_ref in used_hbm_bindings:
                binding = bindings.get(binding_ref)
                if binding is None:
                    raise SchemaError(
                        "GlobalAction state use references an unknown HBM binding",
                        path=f"{path}.fragments",
                    )
                declaration = declarations[binding.state_ref]
                space = spaces[binding.die_id]
                expected_state_by_binding[binding_ref] = StateABI.create(
                    state_ref=declaration.id,
                    hbm_binding_ref=binding.id,
                    kind=declaration.identity.kind,
                    lifetime=declaration.lifetime,
                    access=declaration.access,
                    shape=declaration.shape,
                    dtype=declaration.dtype,
                    layout=declaration.layout,
                    die_id=binding.die_id,
                    address=binding.address,
                    size_bytes=binding.size_bytes,
                    alignment_bytes=space.alignment_bytes,
                )
            if actual_state_by_binding != expected_state_by_binding:
                raise SchemaError(
                    "StateABI tuple must exactly preserve used IR1 declarations, bindings, and home address spaces",
                    path=f"{path}.fragments",
                )
        elif actual_state_by_binding:
            raise SchemaError(
                "lowering fragments cannot invent unused StateABI declarations",
                path=f"{path}.fragments",
            )

        if self.fragments != fragments:
            raise SchemaError("linked fragment tuple must exactly equal linker inputs", path=f"{path}.fragments")
        if (
            self.source_ir1_id != ir1.id
            or self.source_projection_id != projection.id
            or self.source_schedule_set_id != schedule_set.id
            or self.source_global_dag_id != dag.id
        ):
            raise SchemaError("source artifact ids do not exactly identify lowering inputs", path=path)

        from .serde import canonical_digest

        actual_inputs: list[tuple[ManifestInputKind, object]] = [
            (ManifestInputKind.IR1, ir1),
            *((ManifestInputKind.FUSION_PLAN, plan) for plan in fusion_plans),
            *((ManifestInputKind.STANDALONE_PLAN, plan) for plan in standalone_plans),
            (ManifestInputKind.IR2_PROJECTION, projection),
            (ManifestInputKind.SCHEDULE_SET, schedule_set),
            (ManifestInputKind.GLOBAL_ACTION_DAG, dag),
        ]
        for linked in fragments:
            if isinstance(linked, RegionManifest):
                actual_inputs.append((ManifestInputKind.REGION_MANIFEST, linked))
                actual_inputs.append(
                    (ManifestInputKind.COMMAND_FRAGMENT, linked.fragment)
                )
            else:
                actual_inputs.append((ManifestInputKind.COMMAND_FRAGMENT, linked))
        expected_digests = tuple(
            sorted(
                (
                    ManifestInputDigest(
                        kind,
                        artifact.id,
                        artifact.schema_version,
                        canonical_digest(artifact),
                    )
                    for kind, artifact in actual_inputs
                ),
                key=lambda item: (item.kind.value, item.artifact_id),
            )
        )
        if self.input_digests != expected_digests:
            raise SchemaError("input digests must exactly cover every lowering input", path=f"{path}.input_digests")

        leaf_fragments = {
            (linked.fragment.id if isinstance(linked, RegionManifest) else linked.id):
            (linked.fragment if isinstance(linked, RegionManifest) else linked)
            for linked in fragments
        }
        for index, linked in enumerate(fragments):
            linked.validate_against(dag, f"{path}.fragments[{index}]")

        actions = {action.id: action for action in dag.actions}
        recv_by_wait, wait_by_recv = _fused_recv_wait_pairs(actions, path)
        wave_incoming_by_send, wave_outgoing_by_wait = (
            state_transfer_wave_action_maps(actions, path)
        )
        executable = {
            action.id: action
            for action in dag.actions
            if action.task_kind is not SemanticTaskKind.TRANSIT
        }
        owner_by_action: dict[str, str] = {}
        records_by_action: dict[str, list[LinkedRecordRef]] = {}
        for fragment_id, fragment in leaf_fragments.items():
            for action_id in fragment.claimed_action_ids:
                if action_id in owner_by_action:
                    raise SchemaError("executable action is claimed by multiple fragments", path=f"{path}.fragments")
                owner_by_action[action_id] = fragment_id
            for stream in fragment.core_streams:
                for record_index, record in enumerate(stream.records):
                    records_by_action.setdefault(record.source_global_action_id, []).append(
                        LinkedRecordRef(fragment_id, record_index, record.source_global_action_id)
                    )
        if set(owner_by_action) != set(executable):
            raise SchemaError("fragments must cover every executable action exactly once and exclude TRANSIT", path=f"{path}.fragments")

        executable_cores = tuple(
            sorted(
                {action.logical_core for action in executable.values()},
                key=lambda core: (core.die_id, core.local_core_id),
            )
        )
        if tuple(binding.logical_core for binding in self.core_bindings) != executable_cores:
            raise SchemaError("active core bindings must exactly derive from executable actions", path=f"{path}.core_bindings")
        if any(not stream.records for stream in self.core_streams):
            raise SchemaError("MVP executable subset rejects empty linked core streams", path=f"{path}.core_streams")

        core_specs: dict[LogicalCoreRef, object] = {}
        profiles = {profile.id: profile for profile in ir1.fabric.sram_profiles}
        for die in ir1.fabric.dies:
            for core in die.cores:
                core_specs[LogicalCoreRef(die.id, core.local_core_id)] = core
        for index, binding in enumerate(self.core_bindings):
            core = core_specs.get(binding.logical_core)
            if core is None or (
                binding.core_spec_ref,
                binding.runtime_core_id,
                binding.sram_profile_ref,
            ) != (core.id, core.runtime_core_id, core.sram_profile_ref):
                raise SchemaError("logical-to-runtime core binding disagrees with IR-1 fabric", path=f"{path}.core_bindings[{index}]")

        expected_streams: list[LinkedCoreStream] = []
        for binding in self.core_bindings:
            ordered_actions = sorted(
                (
                    action
                    for action in executable.values()
                    if action.logical_core == binding.logical_core
                ),
                key=lambda action: action.core_order_index,
            )
            record_refs = tuple(
                record_ref
                for action in ordered_actions
                for record_ref in records_by_action[action.id]
            )
            expected_streams.append(
                LinkedCoreStream(binding.logical_core, binding.runtime_core_id, record_refs)
            )
        if self.core_streams != tuple(expected_streams):
            raise SchemaError("linked streams must concatenate every fragment record in exact GlobalAction core order", path=f"{path}.core_streams")

        schedules = {schedule.id: schedule for schedule in schedule_set.schedules}
        schedule_bindings = {
            (schedule.id, binding.id): (schedule, binding)
            for schedule in schedule_set.schedules
            for binding in schedule.buffer_bindings
        }
        abi_by_id: dict[str, BufferABI] = {}
        abi_by_schedule_binding: dict[tuple[str, str], BufferABI] = {}
        for fragment in leaf_fragments.values():
            for abi in fragment.buffer_abi:
                abi_by_id.setdefault(abi.id, abi)
                abi_by_schedule_binding.setdefault((abi.schedule_id, abi.binding_id), abi)
        used_schedule_bindings = {
            (action.source.schedule_id, use.binding_id)
            for action in executable.values()
            for use in action.buffer_uses
        }
        if set(abi_by_schedule_binding) != used_schedule_bindings:
            raise SchemaError("BufferABI must exactly cover every scheduled action buffer use", path=f"{path}.fragments")
        for key, abi in abi_by_schedule_binding.items():
            schedule_binding = schedule_bindings.get(key)
            if schedule_binding is None:
                raise SchemaError("BufferABI references an unknown ScheduleSet binding", path=f"{path}.fragments")
            schedule, binding = schedule_binding
            die = next(item for item in ir1.fabric.dies if item.id == schedule.die_id)
            core = next(item for item in die.cores if item.runtime_core_id == binding.core_id)
            expected_core = LogicalCoreRef(die.id, core.local_core_id)
            actual = (
                abi.value_id,
                abi.logical_core,
                abi.tensor_slice,
                abi.region_ref,
                abi.region_offset_bytes,
                abi.size_bytes,
                abi.alignment_bytes,
                abi.banks,
                abi.storage_id,
                abi.alias_of,
                abi.lifetime_start,
                abi.lifetime_end_exclusive,
                abi.dtype,
                abi.layout,
                abi.ownership,
            )
            expected = (
                binding.value_id,
                expected_core,
                binding.tensor_slice,
                binding.region_ref,
                binding.region_offset_bytes,
                binding.size_bytes,
                binding.alignment_bytes,
                binding.banks,
                binding.storage_id,
                binding.alias_of,
                binding.lifetime_start,
                binding.lifetime_end_exclusive,
                binding.dtype,
                binding.layout,
                binding.ownership,
            )
            if actual != expected:
                raise SchemaError("BufferABI does not exactly preserve its ScheduleSet BufferBinding", path=f"{path}.fragments")

        address_bindings = {
            (binding.fragment_id, binding.logical_core, binding.fragment_record_index, binding.operand_id): binding
            for binding in self.address_operand_bindings
        }
        state_bindings = {
            (
                binding.fragment_id,
                binding.logical_core,
                binding.fragment_record_index,
                binding.operand_id,
            ): binding
            for binding in self.state_operand_bindings
        }
        state_abi_by_id = {abi.id: abi for abi in actual_state_by_binding.values()}
        program_definitions = {
            definition.symbol.id: definition
            for definition in self.program_symbol_definitions
        }
        roles_by_operand = {
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.MATMUL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.ATTENTION, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.SWIGLU, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.RESIDUAL, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.ROPE_QK_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.ATTENTION_EXACT, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.EMBEDDING_LOOKUP, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.CROSS_ENTROPY_FORWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_AUX_ADDRESS): (BufferUseRole.COMP_INPUT, 2),
            (RecordOpcode.CROSS_ENTROPY_BACKWARD, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_DATA_ADDRESS): (BufferUseRole.COMP_INPUT, 1),
            (RecordOpcode.SGD_UPDATE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_INPUT_ADDRESS): (BufferUseRole.COMP_INPUT, 0),
            (RecordOpcode.GREEDY_SAMPLE, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.RMSNORM, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS): (BufferUseRole.COMP_OUTPUT, 0),
            (RecordOpcode.DTE_SEND, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.SEND_SOURCE, 0),
            (RecordOpcode.DTE_RECV, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.RECV_DESTINATION, 0),
            (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.REDUCE_INPUT, -1),
            (RecordOpcode.LOCAL_REDUCE, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.REDUCE_OUTPUT, 0),
            (RecordOpcode.DTE_ISSUE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.LOCAL_COPY_SOURCE, 0),
            (RecordOpcode.DTE_ISSUE, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.LOCAL_COPY_DESTINATION, 0),
            (RecordOpcode.LSU_LOAD, SemanticOperandId.DESTINATION_ADDRESS): (BufferUseRole.DMA_DESTINATION, 0),
            (RecordOpcode.LSU_STORE, SemanticOperandId.SOURCE_ADDRESS): (BufferUseRole.DMA_SOURCE, 0),
        }

        def role_for_operand(
            opcode: RecordOpcode,
            operand_id: SemanticOperandId,
        ) -> tuple[BufferUseRole, int]:
            if opcode is RecordOpcode.SRAM_BIND:
                if operand_id is SemanticOperandId.SRAM_BIND_OUTPUT:
                    return (BufferUseRole.COMP_OUTPUT, 0)
                if operand_id in _SRAM_BIND_INPUT_OPERAND_IDS:
                    return (
                        BufferUseRole.COMP_INPUT,
                        _SRAM_BIND_INPUT_OPERAND_IDS.index(operand_id),
                    )
            result = roles_by_operand.get((opcode, operand_id))
            if result is None:
                raise SchemaError(
                    "address operand has no frozen GlobalAction buffer role",
                    path=f"{path}.address_operand_bindings",
                )
            return result

        def lifecycle_abi(
            closure: AddressOperandBinding,
            action: GlobalAction,
            record: RelocatableRecord,
            operand_id: SemanticOperandId,
            symbol_ref: str,
        ) -> BufferABI:
            if len(closure.buffer_abi_ids) != 1:
                raise SchemaError(
                    "lifecycle address closure requires one BufferABI",
                    path=f"{path}.address_operand_bindings",
                )
            abi = abi_by_id.get(closure.buffer_abi_ids[0])
            if abi is None:
                raise SchemaError(
                    "lifecycle address closure references an unknown BufferABI",
                    path=f"{path}.address_operand_bindings",
                )
            if (
                (action.source.schedule_id, abi.binding_id)
                not in {
                    (action.source.schedule_id, use.binding_id)
                    for use in action.buffer_uses
                }
                or abi.logical_core != action.logical_core
                or abi.alias_of is not None
            ):
                raise SchemaError(
                    "lifecycle BufferABI must be a non-aliased use of its owning action/core",
                    path=f"{path}.address_operand_bindings",
                )
            if closure.tensor_slices != (abi.tensor_slice,):
                raise SchemaError(
                    "lifecycle address closure must preserve the exact BufferABI root",
                    path=f"{path}.address_operand_bindings",
                )
            definition = program_definitions.get(symbol_ref)
            if definition is None:
                raise SchemaError(
                    "lifecycle relocation references an unknown program symbol",
                    path=f"{path}.program_symbol_definitions",
                )
            symbol = definition.symbol
            if record.opcode is RecordOpcode.SRAM_ALLOC_AT:
                if abi.lifetime_start != action.core_order_index:
                    raise SchemaError(
                        "SRAM_ALLOC_AT must be owned by the exact first-use action",
                        path=f"{path}.address_operand_bindings",
                    )
                if operand_id is SemanticOperandId.REGION_NAME:
                    expected = (ProgramSymbolKind.SRAM_REGION, abi.region_ref)
                elif operand_id is SemanticOperandId.LABEL_SYMBOL:
                    expected = (ProgramSymbolKind.SRAM_LABEL, abi.storage_id)
                else:
                    raise SchemaError(
                        "SRAM_ALLOC_AT has an invalid semantic operand",
                        path=f"{path}.address_operand_bindings",
                    )
            elif record.opcode is RecordOpcode.SRAM_FREE:
                if (
                    operand_id is not SemanticOperandId.SYMBOL
                    or abi.lifetime_end_exclusive != action.core_order_index + 1
                ):
                    raise SchemaError(
                        "SRAM_FREE must be owned by the exact last-use action",
                        path=f"{path}.address_operand_bindings",
                    )
                expected = (ProgramSymbolKind.SRAM_LABEL, abi.storage_id)
            else:
                raise SchemaError("not a lifecycle record", path=path)
            if (symbol.kind, symbol.source_ref) != expected:
                raise SchemaError(
                    "lifecycle symbol kind/source does not match BufferABI",
                    path=f"{path}.program_symbol_definitions",
                )
            return abi
        symbol_cores: dict[str, set[LogicalCoreRef]] = {}
        relocated_program_symbols: set[str] = set()
        containment_witnesses: set[str] = set()
        lifecycle_uses: dict[
            tuple[str, LogicalCoreRef, str], dict[str, tuple[str, str, int]]
        ] = {}
        lifecycle_enabled = any(
            record.opcode in _LIFECYCLE_OPCODES
            for fragment in leaf_fragments.values()
            for stream in fragment.core_streams
            for record in stream.records
        )
        for fragment_id, fragment in leaf_fragments.items():
            for stream in fragment.core_streams:
                for relocation in stream.address_relocations:
                    record = stream.records[relocation.record_index]
                    action = actions[record.source_global_action_id]
                    closure_key = (
                        fragment_id,
                        stream.logical_core,
                        relocation.record_index,
                        relocation.operand_id,
                    )
                    if relocation.operand_id is SemanticOperandId.HBM_ADDRESS:
                        state_closure = state_bindings.get(closure_key)
                        if state_closure is None:
                            raise SchemaError(
                                "HBM relocation requires its exact StateOperandBinding",
                                path=f"{path}.state_operand_bindings",
                            )
                        abi = state_abi_by_id[state_closure.state_abi_id]
                        state_use = (
                            action.state_uses[0]
                            if len(action.state_uses) == 1
                            else None
                        )
                        definition = program_definitions[relocation.symbol_ref]
                        if (
                            action.dma is None
                            or state_use is None
                            or state_use.hbm_binding_ref != abi.hbm_binding_ref
                            or action.dma.state_ref != abi.state_ref
                            or stream.logical_core.die_id != abi.die_id
                            or definition.symbol.source_ref != abi.hbm_binding_ref
                            or definition.value != abi.address
                            or definition.size_bytes != abi.size_bytes
                            or relocation.addend
                            != action.dma.state_offset_bytes
                        ):
                            raise SchemaError(
                                "HBM relocation does not exactly preserve its GlobalAction and StateABI",
                                path=f"{path}.state_operand_bindings",
                            )
                        if (
                            action.dma.state_offset_bytes > abi.size_bytes
                            or action.bytes
                            > abi.size_bytes - action.dma.state_offset_bytes
                        ):
                            raise SchemaError(
                                "GlobalAction state byte range exceeds StateABI",
                                path=f"{path}.state_operand_bindings",
                            )
                        relocated_program_symbols.add(relocation.symbol_ref)
                        symbol_cores.setdefault(relocation.symbol_ref, set()).add(
                            stream.logical_core
                        )
                        continue
                    closure = address_bindings[closure_key]
                    if record.opcode in _LIFECYCLE_OPCODES:
                        abi = lifecycle_abi(
                            closure,
                            action,
                            record,
                            relocation.operand_id,
                            relocation.symbol_ref,
                        )
                        closure_abis = (abi,)
                        view_addends = (0,)
                        view_lengths = (abi.size_bytes,)
                        key = (abi.schedule_id, abi.logical_core, abi.storage_id)
                        roles = lifecycle_uses.setdefault(key, {})
                        role = (
                            "region"
                            if relocation.operand_id is SemanticOperandId.REGION_NAME
                            else "alloc"
                            if record.opcode is RecordOpcode.SRAM_ALLOC_AT
                            else "free"
                        )
                        occurrence = (
                            relocation.symbol_ref,
                            fragment_id,
                            relocation.record_index,
                        )
                        if role in roles:
                            raise SchemaError(
                                "each SRAM storage requires exactly one lifecycle role",
                                path=f"{path}.fragments",
                            )
                        roles[role] = occurrence
                    else:
                        role, operand_index = role_for_operand(
                            record.opcode, relocation.operand_id
                        )
                        closure_abis, view_addends, view_lengths = _validate_address_operand_closure(
                            closure,
                            action,
                            record,
                            role,
                            operand_index,
                            abi_by_schedule_binding,
                            abi_by_id,
                            f"{path}.address_operand_bindings",
                        )
                    definition = program_definitions[relocation.symbol_ref]
                    relocated_program_symbols.add(relocation.symbol_ref)
                    resolved_regions = []
                    absolute_starts = []
                    absolute_ends = []
                    view_starts = []
                    view_ends = []
                    for abi, view_addend, view_length in zip(
                        closure_abis, view_addends, view_lengths
                    ):
                        core = core_specs[abi.logical_core]
                        profile = profiles[core.sram_profile_ref]
                        region = next((item for item in profile.regions if item.id == abi.region_ref), None)
                        if region is None:
                            raise SchemaError("BufferABI references an unknown IR-1 named SRAM region", path=f"{path}.program_symbol_definitions")
                        resolved_regions.append(region)
                        absolute_start = region.base_bytes + abi.region_offset_bytes
                        absolute_starts.append(absolute_start)
                        absolute_ends.append(absolute_start + abi.size_bytes)
                        view_starts.append(absolute_start + view_addend)
                        view_ends.append(absolute_start + view_addend + view_length)
                        if view_addend + view_length > abi.size_bytes:
                            raise SchemaError(
                                "relocated tensor view exceeds its BufferABI root",
                                path=f"{path}.address_operand_bindings",
                            )
                    if record.opcode is RecordOpcode.LOCAL_REDUCE:
                        _validate_local_reduce_absolute_alignment(
                            tuple(absolute_starts),
                            f"{path}.address_operand_bindings",
                            alignment=(
                                4
                                if record.operands[0].literal_value == 1
                                else 2
                            ),
                        )
                    if definition.symbol.kind is ProgramSymbolKind.SRAM_REGION:
                        expected_region = resolved_regions[0]
                        if any(
                            (region.name, region.base_bytes, region.size_bytes)
                            != (expected_region.name, expected_region.base_bytes, expected_region.size_bytes)
                            for region in resolved_regions
                        ) or (
                            definition.name,
                            definition.value,
                            definition.size_bytes,
                            relocation.addend,
                        ) != (
                            expected_region.name,
                            expected_region.base_bytes,
                            expected_region.size_bytes,
                            (
                                0
                                if record.opcode is RecordOpcode.SRAM_ALLOC_AT
                                else closure_abis[0].region_offset_bytes
                                + view_addends[0]
                            ),
                        ):
                            raise SchemaError("named SRAM region definition/addend is not exact", path=f"{path}.program_symbol_definitions")
                    elif definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
                        root_start = min(absolute_starts)
                        root_end = max(absolute_ends)
                        view_start = min(view_starts)
                        view_end = max(view_ends)
                        if (
                            definition.value != root_start
                            or definition.size_bytes != root_end - root_start
                            or relocation.addend != view_start - root_start
                            or view_end > root_end
                        ):
                            raise SchemaError("absolute address definition/addend does not resolve to BufferABI root and exact tensor view", path=f"{path}.program_symbol_definitions")
                        if record.opcode is RecordOpcode.LOCAL_REDUCE:
                            region = resolved_regions[0]
                            containment_witnesses.add(
                                _validate_local_reduce_containment_witness(
                                    fragment.program_symbols,
                                    program_definitions,
                                    stream.logical_core,
                                    region.name,
                                    region.base_bytes,
                                    region.size_bytes,
                                    view_start,
                                    view_end,
                                    f"{path}.program_symbol_definitions",
                                )
                            )
                    elif (
                        definition.symbol.kind is not ProgramSymbolKind.SRAM_LABEL
                        or record.opcode
                        not in (
                            RecordOpcode.SRAM_BIND,
                            RecordOpcode.SRAM_ALLOC_AT,
                            RecordOpcode.SRAM_FREE,
                        )
                        or definition.value != 0
                        or definition.size_bytes != 0
                        or relocation.addend != 0
                    ):
                        raise SchemaError(
                            "SRAM lifecycle/bind labels must remain non-physical zero definitions",
                            path=f"{path}.program_symbol_definitions",
                        )
                    symbol_cores.setdefault(relocation.symbol_ref, set()).add(stream.logical_core)
        if lifecycle_enabled:
            expected_lifecycle: dict[
                tuple[str, LogicalCoreRef, str], BufferABI
            ] = {}
            roots = _canonical_lifecycle_roots(
                tuple(abi_by_schedule_binding.values()),
                path=f"{path}.fragments",
            )
            for abi in {root.id: root for root in roots.values()}.values():
                key = (abi.schedule_id, abi.logical_core, abi.storage_id)
                previous = expected_lifecycle.setdefault(key, abi)
                if previous != abi:
                    raise SchemaError(
                        "one lifecycle storage has conflicting BufferABI placement",
                        path=f"{path}.fragments",
                    )
            if set(lifecycle_uses) != set(expected_lifecycle):
                raise SchemaError(
                    "lifecycle records must exactly cover every scheduled storage",
                    path=f"{path}.fragments",
                )
            for key, roles in lifecycle_uses.items():
                if set(roles) != {"region", "alloc", "free"}:
                    raise SchemaError(
                        "each SRAM storage requires one ALLOC_AT region/label and one FREE",
                        path=f"{path}.fragments",
                    )
                if roles["alloc"][0] != roles["free"][0]:
                    raise SchemaError(
                        "SRAM_ALLOC_AT and SRAM_FREE must share one storage label",
                        path=f"{path}.program_symbol_definitions",
                    )
        for symbol_ref, cores in symbol_cores.items():
            expected = tuple(sorted(cores, key=lambda core: (core.die_id, core.local_core_id)))
            if program_definitions[symbol_ref].logical_cores != expected:
                raise SchemaError("program symbol logical-core scope is not exact", path=f"{path}.program_symbol_definitions")
        unused_program_symbols = set(program_definitions).difference(
            relocated_program_symbols
        )
        if not unused_program_symbols.issubset(containment_witnesses):
            raise SchemaError(
                "unused program declarations are forbidden except exact LOCAL_REDUCE containment witnesses; a witness may also serve SRAM_ALLOC_AT",
                path=f"{path}.program_symbol_definitions",
            )

        runtime_definitions = {
            definition.symbol.id: definition
            for definition in self.runtime_symbol_definitions
        }
        groups = {group.symbol_ref: group for group in self.core_groups}
        _validate_fused_recv_wait_token_closure(
            actions,
            tuple(leaf_fragments.values()),
            runtime_definitions,
            f"{path}.runtime_symbol_definitions",
        )

        def transport_peer(action: GlobalAction) -> GlobalAction:
            opposite = (
                SemanticTaskKind.RECV
                if action.task_kind is SemanticTaskKind.SEND
                else SemanticTaskKind.SEND
            )
            candidates = [
                candidate
                for candidate in actions.values()
                if candidate.task_kind is opposite
                and candidate.flow_id == action.flow_id
                and candidate.source_rank == action.source_rank
                and candidate.destination_rank == action.destination_rank
            ]
            if len(candidates) != 1:
                raise SchemaError(
                    "transport runtime symbols require one exact SEND/RECV peer",
                    path=f"{path}.runtime_symbol_definitions",
                )
            return candidates[0]

        paired_runtime_symbols: dict[tuple[str, str, RuntimeOperandField], str] = {}
        async_token_records: dict[
            tuple[str, str], dict[SemanticTaskKind, str]
        ] = {}
        dependency_event_counts: dict[
            tuple[str, str, str], dict[RecordOpcode, int]
        ] = {}
        plan_barrier_event_counts: dict[
            tuple[str, str, str], dict[RecordOpcode, int]
        ] = {}
        expected_plan_barrier_events = _expected_plan_barrier_events(dag, path)
        for fragment_id, fragment in leaf_fragments.items():
            for stream in fragment.core_streams:
                for record_index, record in enumerate(stream.records):
                    action = actions[record.source_global_action_id]
                    runtime_by_field = {
                        relocation.field: runtime_definitions[relocation.symbol_ref]
                        for relocation in stream.runtime_relocations
                        if relocation.record_index == record_index
                    }
                    for field, definition in runtime_by_field.items():
                        if field is RuntimeOperandField.DTE_TOKEN:
                            if action.task_kind is SemanticTaskKind.LOCAL_COPY:
                                if definition.symbol.source_ref != action.id:
                                    raise SchemaError(
                                        "LOCAL_COPY token source_ref must be its operation-local action id",
                                        path=f"{path}.runtime_symbol_definitions",
                                    )
                            else:
                                _validate_runtime_binding_ref(
                                    definition,
                                    action,
                                    field,
                                    f"{path}.runtime_symbol_definitions",
                                )
                            recv = (
                                action
                                if action.task_kind is SemanticTaskKind.RECV
                                and action.id in wait_by_recv
                                else recv_by_wait.get(action.id)
                            )
                            wait = wait_by_recv.get(recv.id) if recv is not None else None
                            if recv is not None and wait is not None:
                                exact_endpoints = (
                                    action.task_kind
                                    in (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
                                    and definition.logical_cores == (recv.logical_core,)
                                    and definition.source_action_id == recv.id
                                    and definition.destination_action_id == wait.id
                                )
                                roles = async_token_records.setdefault(
                                    (recv.id, wait.id), {}
                                )
                                previous = roles.setdefault(
                                    action.task_kind, definition.symbol.id
                                )
                                exact_endpoints = (
                                    exact_endpoints
                                    and previous == definition.symbol.id
                                )
                            else:
                                exact_endpoints = (
                                    definition.logical_cores == (stream.logical_core,)
                                    and definition.source_action_id == action.id
                                    and definition.destination_action_id is None
                                )
                            if not exact_endpoints:
                                raise SchemaError(
                                    "DTE token definition must exactly bind its issuer/consumer actions and owning core",
                                    path=f"{path}.runtime_symbol_definitions",
                                )
                        elif field in (
                            RuntimeOperandField.PEER_CORE,
                            RuntimeOperandField.DTE_FSM,
                        ):
                            if action.task_kind not in (
                                SemanticTaskKind.SEND,
                                SemanticTaskKind.RECV,
                            ):
                                raise SchemaError(
                                    "transport runtime field is attached to a non-transport action",
                                    path=f"{path}.runtime_symbol_definitions",
                                )
                            peer = transport_peer(action)
                            _validate_runtime_binding_ref(
                                definition,
                                action,
                                field,
                                f"{path}.runtime_symbol_definitions",
                            )
                            if field is RuntimeOperandField.PEER_CORE:
                                if (
                                    definition.logical_cores != (peer.logical_core,)
                                    or definition.source_action_id is not None
                                    or definition.destination_action_id is not None
                                ):
                                    raise SchemaError(
                                        "peer-core definition does not exactly name the opposite transport core",
                                        path=f"{path}.runtime_symbol_definitions",
                                    )
                            else:
                                send, recv = (
                                    (action, peer)
                                    if action.task_kind is SemanticTaskKind.SEND
                                    else (peer, action)
                                )
                                pair = (send.id, recv.id)
                                expected_cores = tuple(
                                    sorted(
                                        (send.logical_core, recv.logical_core),
                                        key=lambda core: (
                                            core.die_id,
                                            core.local_core_id,
                                        ),
                                    )
                                )
                                if (
                                    definition.logical_cores != expected_cores
                                    or definition.source_action_id != send.id
                                    or definition.destination_action_id != recv.id
                                ):
                                    raise SchemaError(
                                        "DTE FSM definition does not exactly bind one SEND/RECV pair",
                                        path=f"{path}.runtime_symbol_definitions",
                                    )
                                key = (*pair, field)
                                previous = paired_runtime_symbols.setdefault(
                                    key, definition.symbol.id
                                )
                                if previous != definition.symbol.id:
                                    raise SchemaError(
                                        "paired SEND/RECV records must share one DTE FSM symbol",
                                        path=f"{path}.runtime_symbol_definitions",
                                    )
                        elif field is RuntimeOperandField.GROUP_ID:
                            group = groups.get(definition.symbol.id)
                            if group is None or stream.logical_core not in group.members:
                                raise SchemaError(
                                    "GROUP relocation does not resolve to a same-die group containing the executing core",
                                    path=f"{path}.core_groups",
                                )

                    if record.opcode in (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT):
                        required_fields = {
                            RuntimeOperandField.SOURCE_CORE,
                            RuntimeOperandField.DESTINATION_CORE,
                            RuntimeOperandField.EVENT_TAG,
                        }
                        if set(runtime_by_field) != required_fields:
                            raise SchemaError(
                                "event record requires exact source/destination/tag relocations",
                                path=f"{path}.runtime_symbol_definitions",
                            )
                        event = runtime_by_field[RuntimeOperandField.EVENT_TAG]
                        source = actions.get(event.source_action_id or "")
                        destination = actions.get(event.destination_action_id or "")
                        if (
                            source is None
                            or destination is None
                            or source.logical_core is None
                            or destination.logical_core is None
                            or source.logical_core == destination.logical_core
                        ):
                            raise SchemaError(
                                "event definition endpoints must be two executable actions on different cores",
                                path=f"{path}.runtime_symbol_definitions",
                            )
                        expected_event_cores = tuple(
                            sorted(
                                (source.logical_core, destination.logical_core),
                                key=lambda core: (core.die_id, core.local_core_id),
                            )
                        )
                        key = (source.id, destination.id, event.symbol.id)
                        plan_spec = expected_plan_barrier_events.get(key)
                        wave_event = (
                            wave_incoming_by_send.get(destination.id) == source
                            and wave_outgoing_by_wait.get(source.id)
                            == destination
                        )
                        if not wave_event:
                            for field in required_fields:
                                _validate_runtime_binding_ref(
                                    runtime_by_field[field],
                                    source,
                                    field,
                                    f"{path}.runtime_symbol_definitions",
                                )
                            _validate_runtime_binding_ref(
                                event,
                                destination,
                                RuntimeOperandField.EVENT_TAG,
                                f"{path}.runtime_symbol_definitions",
                            )
                        source_core_definition = runtime_by_field[
                            RuntimeOperandField.SOURCE_CORE
                        ]
                        destination_core_definition = runtime_by_field[
                            RuntimeOperandField.DESTINATION_CORE
                        ]
                        common_is_exact = (
                            event.logical_cores == expected_event_cores
                            and source_core_definition.logical_cores
                            == (source.logical_core,)
                            and destination_core_definition.logical_cores
                            == (destination.logical_core,)
                            and (
                                record.opcode is RecordOpcode.EVENT_SET
                                and action.id == source.id
                                or record.opcode is RecordOpcode.EVENT_WAIT
                                and action.id == destination.id
                            )
                        )
                        if plan_spec is not None:
                            expected_source_core = canonical_plan_barrier_core_symbol(
                                dag.id, source
                            )
                            expected_destination_core = (
                                canonical_plan_barrier_core_symbol(
                                    dag.id, destination
                                )
                            )
                            if (
                                not common_is_exact
                                or plan_spec.source != source
                                or plan_spec.destination != destination
                                or plan_spec.event != event.symbol
                                or source_core_definition.symbol
                                != expected_source_core
                                or destination_core_definition.symbol
                                != expected_destination_core
                                or source.logical_core.die_id
                                == destination.logical_core.die_id
                            ):
                                raise SchemaError(
                                    "PLAN barrier event definitions must exactly preserve canonical coordinator direction and symbols",
                                    path=f"{path}.runtime_symbol_definitions",
                                )
                            counts = plan_barrier_event_counts.setdefault(key, {})
                        elif wave_event:
                            (
                                expected_source_core,
                                expected_destination_core,
                                expected_event,
                            ) = canonical_state_transfer_wave_symbols(
                                dag.id, source, destination
                            )
                            if (
                                not common_is_exact
                                or source_core_definition.symbol
                                != expected_source_core
                                or destination_core_definition.symbol
                                != expected_destination_core
                                or event.symbol != expected_event
                            ):
                                raise SchemaError(
                                    "state-transfer wave event definitions must exactly preserve canonical WAIT-to-SEND direction and symbols",
                                    path=f"{path}.runtime_symbol_definitions",
                                )
                            counts = dependency_event_counts.setdefault(
                                key, {}
                            )
                        else:
                            if (
                                not common_is_exact
                                or source.sync is None
                                or destination.sync is None
                                or source.sync.completion_event
                                != event.symbol.source_ref
                                or destination.sync.wait_event
                                != event.symbol.source_ref
                            ):
                                raise SchemaError(
                                    "event record runtime definitions do not preserve dependency direction",
                                    path=f"{path}.runtime_symbol_definitions",
                                )
                            counts = dependency_event_counts.setdefault(key, {})
                        counts[record.opcode] = counts.get(record.opcode, 0) + 1

        expected_async_pairs = {
            (recv_id, wait.id) for recv_id, wait in wait_by_recv.items()
        }
        if set(async_token_records) != expected_async_pairs or any(
            set(roles) != {SemanticTaskKind.RECV, SemanticTaskKind.WAIT}
            or len(set(roles.values())) != 1
            for roles in async_token_records.values()
        ):
            raise SchemaError(
                "every waited fused RECV/WAIT pair must share one exact DTE token relocation",
                path=f"{path}.runtime_symbol_definitions",
            )

        required_event_pairs = {
            (dependency, action.id)
            for action in executable.values()
            for dependency in action.deps
            if dependency in executable
            and executable[dependency].logical_core != action.logical_core
        }
        actual_event_pairs = {
            (source, destination)
            for source, destination, _symbol in dependency_event_counts
        }
        exact_event_counts = all(
            counts.get(RecordOpcode.EVENT_SET, 0) == 1
            and counts.get(RecordOpcode.EVENT_WAIT, 0) == 1
            and len(counts) == 2
            for counts in dependency_event_counts.values()
        )
        if (
            actual_event_pairs != required_event_pairs
            or len(dependency_event_counts) != len(required_event_pairs)
            or not exact_event_counts
        ):
            raise SchemaError(
                "one exact EVENT SET/WAIT symbol pair must close every cross-core GlobalAction dependency",
                path=f"{path}.fragment_interfaces",
            )
        exact_plan_barrier_counts = all(
            counts.get(RecordOpcode.EVENT_SET, 0) == 1
            and counts.get(RecordOpcode.EVENT_WAIT, 0) == 1
            and len(counts) == 2
            for counts in plan_barrier_event_counts.values()
        )
        if (
            set(plan_barrier_event_counts) != set(expected_plan_barrier_events)
            or not exact_plan_barrier_counts
        ):
            raise SchemaError(
                "one exact coordinator EVENT SET/WAIT pair must close every PLAN barrier ARRIVE and RELEASE",
                path=f"{path}.fragment_interfaces",
            )

        active = executable_cores
        expected_start_targets = tuple(event.target_core for event in self.envelope.start_events)
        first_action_by_core = {
            core: min(
                (
                    action
                    for action in executable.values()
                    if action.logical_core == core
                ),
                key=lambda action: action.core_order_index,
            )
            for core in active
        }
        if (
            self.envelope.active_cores != active
            or self.envelope.terminal_cores != active
            or self.envelope.expected_ack_cores != active
            or self.envelope.expected_done_cores != active
            or self.envelope.empty_core_ack_policy is not EmptyCoreAckPolicy.INCLUDE_EMPTY
            or expected_start_targets != active
            or any(event.count != 1 for event in self.envelope.start_events)
            or len({event.tag_symbol_ref for event in self.envelope.start_events}) != len(active)
            or any(
                runtime_definitions[event.tag_symbol_ref].symbol.source_ref
                != first_action_by_core[event.target_core].id
                for event in self.envelope.start_events
            )
        ):
            raise SchemaError("MVP control envelope requires one independent count=1 START and ACK/DONE for every executable core", path=f"{path}.envelope")
