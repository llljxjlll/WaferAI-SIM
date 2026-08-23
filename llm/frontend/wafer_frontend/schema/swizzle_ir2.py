"""Executable timing-only per-Die projection of a Swizzle fusion plan.

The existing IR2 ``SemanticTask`` contract requires full compute/reduction
contracts that W7 intentionally does not invent.  This isolated carrier keeps
the complete action DAG, temporary lineage, buffers and routes losslessly and
admits only the exact strict Swizzle timing consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    stable_artifact_id,
    validate_dependency_dag,
    validate_nonempty,
    validate_uint64,
    validate_unique_ids,
)
from .ir0 import FusionPattern
from .swizzle import SwizzleActionKind, SwizzleAlgorithm, SwizzlePhase, SwizzleTensorAxis


SWIZZLE_IR2_SCHEMA_VERSION = "wafer_frontend.swizzle_ir2/v1alpha1"
SWIZZLE_IR2_TASK_SCHEMA_VERSION = "wafer_frontend.swizzle_ir2_task/v1alpha1"
SWIZZLE_IR2_VALUE_SCHEMA_VERSION = "wafer_frontend.swizzle_ir2_value/v1alpha1"


class SwizzleIr2ValueOriginKind(str, Enum):
    BOUNDARY_INPUT = "boundary_input"
    LOCAL_OPERAND = "local_operand"
    BUFFER = "buffer"
    ACTION_OUTPUT = "action_output"
    RECEIVED_PAYLOAD = "received_payload"
    LOOP_ACCUMULATOR = "loop_accumulator"


class SwizzleIr2BufferRole(str, Enum):
    TEMPORARY = "temporary"
    DOUBLE_BUFFER = "double_buffer"
    LOOP_ACCUMULATOR = "loop_accumulator"
    REDUCTION = "reduction"
    REPLICATION = "replication"


class SwizzleIr2BufferAccess(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    LIFETIME = "lifetime"


class SwizzleIr2ArStage(str, Enum):
    NONE = "none"
    REDUCTION = "reduction"
    REPLICATION = "replication"


class SwizzleIr2ConsumerContract(str, Enum):
    STRICT_SWIZZLE_TIMING_V1 = "strict_swizzle_timing_v1"
    CURRENT_NAIVE_IR2 = "current_naive_ir2"


@dataclass(frozen=True, slots=True)
class SwizzleIr2TaskBufferUse:
    buffer_ref: str
    access: SwizzleIr2BufferAccess
    slot: int

    def validate(self, path: str = "swizzle_ir2_buffer_use") -> None:
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        if type(self.access) is not SwizzleIr2BufferAccess:
            raise SchemaError("must be a SwizzleIr2BufferAccess", path=f"{path}.access")
        validate_uint64(self.slot, f"{path}.slot")


@dataclass(frozen=True, slots=True)
class SwizzleIr2Value:
    schema_version: str
    id: str
    rank: int
    symbolic_ref: str
    origin_kind: SwizzleIr2ValueOriginKind
    origin_ref: str
    producer_task_refs: tuple[str, ...]
    consumer_task_refs: tuple[str, ...]
    buffer_ref: str | None
    loop_carried: bool

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleIr2Value":
        result = cls(
            schema_version=SWIZZLE_IR2_VALUE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_ir2_value",
                {"rank": semantic["rank"], "symbolic_ref": semantic["symbolic_ref"]},
                schema_version=SWIZZLE_IR2_VALUE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "rank",
                "symbolic_ref",
                "origin_kind",
                "origin_ref",
                "producer_task_refs",
                "consumer_task_refs",
                "buffer_ref",
                "loop_carried",
            )
        }

    def validate(self, path: str = "swizzle_ir2_value") -> None:
        if self.schema_version != SWIZZLE_IR2_VALUE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.symbolic_ref, f"{path}.symbolic_ref")
        if type(self.origin_kind) is not SwizzleIr2ValueOriginKind:
            raise SchemaError("must be a SwizzleIr2ValueOriginKind", path=f"{path}.origin_kind")
        validate_nonempty(self.origin_ref, f"{path}.origin_ref")
        for name in ("producer_task_refs", "consumer_task_refs"):
            refs = getattr(self, name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate task refs", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        if self.buffer_ref is not None:
            validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        if type(self.loop_carried) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.loop_carried")
        if self.origin_kind in (
            SwizzleIr2ValueOriginKind.BOUNDARY_INPUT,
            SwizzleIr2ValueOriginKind.LOCAL_OPERAND,
        ) and self.producer_task_refs:
            raise SchemaError("external value cannot have projected producers", path=f"{path}.producer_task_refs")
        if self.origin_kind in (
            SwizzleIr2ValueOriginKind.BUFFER,
            SwizzleIr2ValueOriginKind.LOOP_ACCUMULATOR,
        ) and self.buffer_ref is None:
            raise SchemaError("buffer-backed value requires buffer_ref", path=f"{path}.buffer_ref")
        if self.origin_kind in (
            SwizzleIr2ValueOriginKind.ACTION_OUTPUT,
            SwizzleIr2ValueOriginKind.RECEIVED_PAYLOAD,
            SwizzleIr2ValueOriginKind.LOOP_ACCUMULATOR,
        ) and not self.producer_task_refs:
            raise SchemaError("projected value requires a producer", path=f"{path}.producer_task_refs")
        if self.loop_carried and self.origin_kind is not SwizzleIr2ValueOriginKind.LOOP_ACCUMULATOR:
            raise SchemaError("loop_carried requires LOOP_ACCUMULATOR origin", path=f"{path}.loop_carried")
        expected = stable_artifact_id(
            "swizzle_ir2_value",
            {"rank": self.rank, "symbolic_ref": self.symbolic_ref},
            schema_version=SWIZZLE_IR2_VALUE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleIr2Buffer:
    rank: int
    buffer_ref: str
    role: SwizzleIr2BufferRole
    size_bytes: int
    slot_count: int
    lifetime_task_refs: tuple[str, ...]
    value_refs: tuple[str, ...]

    def validate(self, path: str = "swizzle_ir2_buffer") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        if type(self.role) is not SwizzleIr2BufferRole:
            raise SchemaError("must be a SwizzleIr2BufferRole", path=f"{path}.role")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.size_bytes")
        validate_uint64(self.slot_count, f"{path}.slot_count")
        if self.slot_count not in (1, 2):
            raise SchemaError("V1 supports one or two slots", path=f"{path}.slot_count")
        if (self.role is SwizzleIr2BufferRole.DOUBLE_BUFFER) != (self.slot_count == 2):
            raise SchemaError("DOUBLE_BUFFER must exactly accompany two slots", path=f"{path}.slot_count")
        for name in ("lifetime_task_refs", "value_refs"):
            refs = getattr(self, name)
            if not refs or len(set(refs)) != len(refs):
                raise SchemaError("must contain unique refs", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleIr2Task:
    schema_version: str
    id: str
    source_action_ref: str
    rank: int
    die_id: int
    kind: SwizzleActionKind
    phase: SwizzlePhase
    ar_stage: SwizzleIr2ArStage
    member_ref: str
    chunk_index: int | None
    deps: tuple[str, ...]
    read_value_refs: tuple[str, ...]
    write_value_refs: tuple[str, ...]
    buffer_uses: tuple[SwizzleIr2TaskBufferUse, ...]
    peer_rank: int | None
    route_ref: str | None
    expected_route: tuple[int, ...]
    logical_bytes: int
    flops: int

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleIr2Task":
        result = cls(
            schema_version=SWIZZLE_IR2_TASK_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_ir2_task",
                semantic,
                schema_version=SWIZZLE_IR2_TASK_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_action_ref",
                "rank",
                "die_id",
                "kind",
                "phase",
                "ar_stage",
                "member_ref",
                "chunk_index",
                "deps",
                "read_value_refs",
                "write_value_refs",
                "buffer_uses",
                "peer_rank",
                "route_ref",
                "expected_route",
                "logical_bytes",
                "flops",
            )
        }

    def validate(self, path: str = "swizzle_ir2_task") -> None:
        if self.schema_version != SWIZZLE_IR2_TASK_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.source_action_ref, f"{path}.source_action_ref")
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if type(self.kind) is not SwizzleActionKind:
            raise SchemaError("must be a SwizzleActionKind", path=f"{path}.kind")
        if type(self.phase) is not SwizzlePhase:
            raise SchemaError("must be a SwizzlePhase", path=f"{path}.phase")
        if type(self.ar_stage) is not SwizzleIr2ArStage:
            raise SchemaError("must be a SwizzleIr2ArStage", path=f"{path}.ar_stage")
        validate_nonempty(self.member_ref, f"{path}.member_ref")
        if self.chunk_index is not None:
            validate_uint64(self.chunk_index, f"{path}.chunk_index")
        for name in ("deps", "read_value_refs", "write_value_refs"):
            refs = getattr(self, name)
            if len(set(refs)) != len(refs):
                raise SchemaError("contains duplicate refs", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        buffer_refs: set[str] = set()
        for index, use in enumerate(self.buffer_uses):
            use.validate(f"{path}.buffer_uses[{index}]")
            if use.buffer_ref in buffer_refs:
                raise SchemaError("duplicate buffer use", path=f"{path}.buffer_uses[{index}]")
            buffer_refs.add(use.buffer_ref)
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        if self.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            if self.peer_rank is None or self.route_ref is None or not self.expected_route or self.logical_bytes == 0 or self.flops != 0:
                raise SchemaError("transport task requires peer/route/payload", path=path)
        else:
            if self.peer_rank is not None or self.route_ref is not None or self.expected_route:
                raise SchemaError("non-transport task cannot carry route fields", path=path)
        if self.kind is SwizzleActionKind.COMP:
            if self.flops == 0 or self.logical_bytes != 0:
                raise SchemaError("COMP requires FLOPs and no byte payload", path=path)
        elif self.flops != 0:
            raise SchemaError("only COMP can carry FLOPs", path=f"{path}.flops")
        expected = stable_artifact_id(
            "swizzle_ir2_task",
            self._semantic_key(),
            schema_version=SWIZZLE_IR2_TASK_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleIr2Flow:
    id: str
    route_ref: str
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    die_path: tuple[int, ...]
    send_task_ref: str
    recv_task_ref: str
    chunk_index: int | None
    logical_bytes: int

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleIr2Flow":
        result = cls(
            id=stable_artifact_id("swizzle_ir2_flow", semantic, schema_version=SWIZZLE_IR2_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "route_ref", "source_rank", "destination_rank", "source_die",
            "destination_die", "die_path", "send_task_ref", "recv_task_ref",
            "chunk_index", "logical_bytes",
        )}

    def validate(self, path: str = "swizzle_ir2_flow") -> None:
        validate_nonempty(self.route_ref, f"{path}.route_ref")
        for name in ("source_rank", "destination_rank", "source_die", "destination_die"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_rank == self.destination_rank or self.source_die == self.destination_die:
            raise SchemaError("flow endpoints must differ", path=path)
        if len(self.die_path) < 2 or self.die_path[0] != self.source_die or self.die_path[-1] != self.destination_die:
            raise SchemaError("die_path must close physical endpoints", path=f"{path}.die_path")
        for name in ("send_task_ref", "recv_task_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.chunk_index is not None:
            validate_uint64(self.chunk_index, f"{path}.chunk_index")
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        if self.logical_bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.logical_bytes")
        expected = stable_artifact_id("swizzle_ir2_flow", self._semantic_key(), schema_version=SWIZZLE_IR2_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleIr2OutputOwnership:
    rank: int
    logical_owner_rank: int
    physical_owner_rank: int
    boundary_output_refs: tuple[str, ...]
    terminal_task_refs: tuple[str, ...]
    replicated: bool

    def validate(self, path: str = "swizzle_ir2_output_ownership") -> None:
        for name in ("rank", "logical_owner_rank", "physical_owner_rank"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if not (self.rank == self.logical_owner_rank == self.physical_owner_rank):
            raise SchemaError("V1 requires identity output ownership", path=path)
        if not self.boundary_output_refs or len(set(self.boundary_output_refs)) != len(self.boundary_output_refs):
            raise SchemaError("must contain unique refs", path=f"{path}.boundary_output_refs")
        if len(set(self.terminal_task_refs)) != len(self.terminal_task_refs):
            raise SchemaError("must contain unique refs", path=f"{path}.terminal_task_refs")
        for name in ("boundary_output_refs", "terminal_task_refs"):
            for index, ref in enumerate(getattr(self, name)):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        if type(self.replicated) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.replicated")


@dataclass(frozen=True, slots=True)
class SwizzleIr2RankDag:
    rank: int
    die_id: int
    tasks: tuple[SwizzleIr2Task, ...]
    values: tuple[SwizzleIr2Value, ...]
    buffers: tuple[SwizzleIr2Buffer, ...]

    def validate(self, path: str = "swizzle_ir2_rank_dag") -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if not self.tasks:
            raise SchemaError("must contain tasks", path=f"{path}.tasks")
        for index, task in enumerate(self.tasks):
            task.validate(f"{path}.tasks[{index}]")
            if task.rank != self.rank or task.die_id != self.die_id:
                raise SchemaError("task placement disagrees with rank DAG", path=f"{path}.tasks[{index}]")
        validate_unique_ids(self.values, f"{path}.values")
        for index, value in enumerate(self.values):
            value.validate(f"{path}.values[{index}]")
            if value.rank != self.rank:
                raise SchemaError("value rank mismatch", path=f"{path}.values[{index}].rank")
        buffer_refs: set[str] = set()
        for index, buffer in enumerate(self.buffers):
            buffer.validate(f"{path}.buffers[{index}]")
            if buffer.rank != self.rank or buffer.buffer_ref in buffer_refs:
                raise SchemaError("buffer rank/ref mismatch", path=f"{path}.buffers[{index}]")
            buffer_refs.add(buffer.buffer_ref)


@dataclass(frozen=True, slots=True)
class SwizzleIr2DownstreamGate:
    current_ir2_compatible: bool
    required_consumer: SwizzleIr2ConsumerContract
    timing_execution: bool
    functional_execution: bool
    reason: str

    def validate(self, path: str = "swizzle_ir2_downstream_gate") -> None:
        if self.current_ir2_compatible:
            raise SchemaError("V1 must not claim current IR2 compatibility", path=f"{path}.current_ir2_compatible")
        if self.required_consumer is not SwizzleIr2ConsumerContract.STRICT_SWIZZLE_TIMING_V1:
            raise SchemaError("requires exact strict Swizzle timing consumer", path=f"{path}.required_consumer")
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("V1 is timing-only", path=path)
        validate_nonempty(self.reason, f"{path}.reason")


@dataclass(frozen=True, slots=True)
class SwizzleIr2Projection:
    schema_version: str
    id: str
    source_adapter_ref: str
    source_ir1_id: str
    source_decision_ref: str
    source_candidate_ref: str
    split_axis: SwizzleTensorAxis
    chunk_count: int
    unroll_degree: int
    fused_op_id: str
    group_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    rank_dags: tuple[SwizzleIr2RankDag, ...]
    flows: tuple[SwizzleIr2Flow, ...]
    output_ownership: tuple[SwizzleIr2OutputOwnership, ...]
    downstream_gate: SwizzleIr2DownstreamGate

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleIr2Projection":
        result = cls(
            schema_version=SWIZZLE_IR2_SCHEMA_VERSION,
            id=stable_artifact_id("swizzle_ir2", semantic, schema_version=SWIZZLE_IR2_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_adapter_ref", "source_decision_ref", "source_candidate_ref",
            "source_ir1_id", "fused_op_id", "group_ref", "pattern", "algorithm",
            "split_axis", "chunk_count", "unroll_degree", "rank_dags", "flows",
            "output_ownership", "downstream_gate",
        )}

    def validate(self, path: str = "swizzle_ir2") -> None:
        if self.schema_version != SWIZZLE_IR2_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("source_adapter_ref", "source_decision_ref", "source_candidate_ref", "source_ir1_id", "fused_op_id", "group_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.pattern) is not FusionPattern or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must use typed pattern/algorithm", path=path)
        if type(self.split_axis) is not SwizzleTensorAxis:
            raise SchemaError("must use a typed split axis", path=f"{path}.split_axis")
        self.split_axis.validate(f"{path}.split_axis")
        validate_uint64(self.chunk_count, f"{path}.chunk_count")
        validate_uint64(self.unroll_degree, f"{path}.unroll_degree")
        if self.chunk_count == 0 or self.unroll_degree not in (1, 2):
            raise SchemaError("requires positive chunks and unroll degree 1 or 2", path=path)
        if tuple(dag.rank for dag in self.rank_dags) != tuple(range(len(self.rank_dags))):
            raise SchemaError("rank DAGs must be dense canonical ranks", path=f"{path}.rank_dags")
        all_tasks: tuple[SwizzleIr2Task, ...] = ()
        all_values: dict[str, SwizzleIr2Value] = {}
        all_buffers: dict[tuple[int, str], SwizzleIr2Buffer] = {}
        for index, dag in enumerate(self.rank_dags):
            dag.validate(f"{path}.rank_dags[{index}]")
            all_tasks += dag.tasks
            for value in dag.values:
                if value.id in all_values:
                    raise SchemaError("value ids must be projection-global", path=f"{path}.rank_dags[{index}].values")
                all_values[value.id] = value
            for buffer in dag.buffers:
                all_buffers[(buffer.rank, buffer.buffer_ref)] = buffer
        task_index = validate_dependency_dag(all_tasks, f"{path}.rank_dags.tasks")
        if len({task.source_action_ref for task in all_tasks}) != len(all_tasks):
            raise SchemaError("source actions must map one-to-one to tasks", path=f"{path}.rank_dags.tasks")
        for task in all_tasks:
            if task.chunk_index is not None and task.chunk_index >= self.chunk_count:
                raise SchemaError("task chunk_index exceeds candidate chunk_count", path=f"{path}.rank_dags.tasks")
            for value_ref in task.read_value_refs + task.write_value_refs:
                value = all_values.get(value_ref)
                if value is None or value.rank != task.rank:
                    raise SchemaError("task value ref is not rank-local and closed", path=f"{path}.rank_dags.tasks")
            for use in task.buffer_uses:
                buffer = all_buffers.get((task.rank, use.buffer_ref))
                if buffer is None or use.slot >= buffer.slot_count:
                    raise SchemaError("task buffer use is not closed", path=f"{path}.rank_dags.tasks")
        for value_ref, value in all_values.items():
            expected_producers = tuple(
                task.id
                for task in all_tasks
                if task.rank == value.rank and value_ref in task.write_value_refs
            )
            expected_consumers = tuple(
                task.id
                for task in all_tasks
                if task.rank == value.rank and value_ref in task.read_value_refs
            )
            if (
                value.producer_task_refs != expected_producers
                or value.consumer_task_refs != expected_consumers
            ):
                raise SchemaError("value producer/consumer incidence is not exact", path=f"{path}.rank_dags.values")
        for (rank, buffer_ref), buffer in all_buffers.items():
            expected_values = tuple(
                value.id
                for value in all_values.values()
                if value.rank == rank and value.buffer_ref == buffer_ref
            )
            expected_lifetime = tuple(
                task.id
                for task in all_tasks
                if task.rank == rank
                and any(use.buffer_ref == buffer_ref for use in task.buffer_uses)
            )
            if buffer.value_refs != expected_values or buffer.lifetime_task_refs != expected_lifetime:
                raise SchemaError("buffer value/lifetime incidence is not exact", path=f"{path}.rank_dags.buffers")
        flow_index = validate_unique_ids(self.flows, f"{path}.flows")
        del flow_index
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            send = task_index.get(flow.send_task_ref)
            recv = task_index.get(flow.recv_task_ref)
            if send is None or recv is None or send.kind is not SwizzleActionKind.SEND or recv.kind is not SwizzleActionKind.RECV:
                raise SchemaError("flow must bind SEND and RECV tasks", path=f"{path}.flows[{index}]")
            if (
                send.rank,
                recv.rank,
                send.die_id,
                recv.die_id,
                send.peer_rank,
                recv.peer_rank,
                send.route_ref,
                recv.route_ref,
                send.expected_route,
                recv.expected_route,
                send.chunk_index,
                recv.chunk_index,
                send.logical_bytes,
                recv.logical_bytes,
            ) != (
                flow.source_rank, flow.destination_rank,
                flow.source_die, flow.destination_die,
                flow.destination_rank, flow.source_rank,
                flow.route_ref, flow.route_ref,
                flow.die_path, flow.die_path,
                flow.chunk_index, flow.chunk_index,
                flow.logical_bytes, flow.logical_bytes,
            ) or send.id not in recv.deps:
                raise SchemaError("flow disagrees with endpoint tasks", path=f"{path}.flows[{index}]")
        send_refs = tuple(flow.send_task_ref for flow in self.flows)
        recv_refs = tuple(flow.recv_task_ref for flow in self.flows)
        expected_sends = tuple(task.id for task in all_tasks if task.kind is SwizzleActionKind.SEND)
        expected_recvs = tuple(task.id for task in all_tasks if task.kind is SwizzleActionKind.RECV)
        if (
            len(set(send_refs)) != len(send_refs)
            or len(set(recv_refs)) != len(recv_refs)
            or set(send_refs) != set(expected_sends)
            or set(recv_refs) != set(expected_recvs)
        ):
            raise SchemaError("flows must cover every SEND/RECV task exactly once", path=f"{path}.flows")
        if tuple(item.rank for item in self.output_ownership) != tuple(range(len(self.rank_dags))):
            raise SchemaError("output ownership must cover every rank", path=f"{path}.output_ownership")
        depended = {dependency for task in all_tasks for dependency in task.deps}
        for index, ownership in enumerate(self.output_ownership):
            ownership.validate(f"{path}.output_ownership[{index}]")
            if any(ref not in task_index for ref in ownership.terminal_task_refs):
                raise SchemaError("ownership references unknown terminal task", path=f"{path}.output_ownership[{index}]")
            expected_terminals = tuple(
                task.id
                for task in all_tasks
                if task.rank == ownership.rank and task.id not in depended
            )
            if ownership.terminal_task_refs != expected_terminals:
                raise SchemaError("ownership terminal tasks are not globally exact", path=f"{path}.output_ownership[{index}]")
            if ownership.replicated != (self.pattern is FusionPattern.GEMM_AR):
                raise SchemaError("replication flag must exactly match GEMM_AR", path=f"{path}.output_ownership[{index}].replicated")
        if self.pattern is FusionPattern.GEMM_AR:
            stages = {task.ar_stage for task in all_tasks}
            if not {SwizzleIr2ArStage.REDUCTION, SwizzleIr2ArStage.REPLICATION}.issubset(stages):
                raise SchemaError("GEMM_AR must retain reduction and replication stages", path=f"{path}.rank_dags.tasks")
            reduction_refs = {
                task.id for task in all_tasks if task.ar_stage is SwizzleIr2ArStage.REDUCTION
            }
            ancestor_cache: dict[str, set[str]] = {}
            def ancestors(task_ref: str) -> set[str]:
                if task_ref not in ancestor_cache:
                    direct = set(task_index[task_ref].deps)
                    transitive = set(direct)
                    for dependency in direct:
                        transitive.update(ancestors(dependency))
                    ancestor_cache[task_ref] = transitive
                return ancestor_cache[task_ref]
            if any(
                task.ar_stage is SwizzleIr2ArStage.REPLICATION
                and not (ancestors(task.id) & reduction_refs)
                for task in all_tasks
            ):
                raise SchemaError("GEMM_AR replication must transitively depend on reduction", path=f"{path}.rank_dags.tasks")
        elif any(task.ar_stage is not SwizzleIr2ArStage.NONE for task in all_tasks):
            raise SchemaError("non-AR task cannot carry AR stage", path=f"{path}.rank_dags.tasks")
        self.downstream_gate.validate(f"{path}.downstream_gate")
        expected = stable_artifact_id("swizzle_ir2", self._semantic_key(), schema_version=SWIZZLE_IR2_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def require_consumer(self, consumer: SwizzleIr2ConsumerContract) -> None:
        if consumer is not self.downstream_gate.required_consumer:
            raise SchemaError(
                f"consumer must be {self.downstream_gate.required_consumer.value!r}",
                path="swizzle_ir2.consumer",
            )


def admits_wang_4rank_packed_layout(
    projection: SwizzleIr2Projection,
) -> bool:
    """Admit packed storage only for the exact proven four-rank Wang cases."""

    return (
        type(projection) is SwizzleIr2Projection
        and projection.algorithm is SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL
        and projection.pattern in (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS)
        and len(projection.rank_dags) == 4
    )


__all__ = [
    name
    for name in globals()
    if name.startswith("SwizzleIr2") or name.startswith("SWIZZLE_IR2")
] + ["admits_wang_4rank_packed_layout"]
