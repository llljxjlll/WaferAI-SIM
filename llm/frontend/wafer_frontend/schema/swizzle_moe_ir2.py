"""Typed MoE Swizzle IR2 used by the multi-core W8/W9 lowering path.

This carrier is deliberately independent from the V1 dense Swizzle schema.  In
particular, every task carries the expert/tile/packet provenance needed to make
a deterministic physical placement without parsing symbolic names.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .swizzle import SwizzleActionKind


MOE_SWIZZLE_IR2_SCHEMA_VERSION = "wafer_frontend.moe_swizzle_ir2/v1alpha1"


def _dtype_bytes(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype in (DType.FP32, DType.INT32):
        return 4
    raise SchemaError("unsupported MoE Swizzle dtype", path="dtype")



def _unique_nonempty(refs: tuple[str, ...], path: str, *, allow_empty: bool = True) -> None:
    if type(refs) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    if not allow_empty and not refs:
        raise SchemaError("must not be empty", path=path)
    if len(refs) != len(set(refs)):
        raise SchemaError("contains duplicate refs", path=path)
    for index, ref in enumerate(refs):
        validate_nonempty(ref, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2BufferUse:
    buffer_ref: str
    slot: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        validate_uint64(self.slot, f"{path}.slot")
        if self.slot not in (0, 1):
            raise SchemaError("MoE Swizzle only admits physical slot 0/1", path=f"{path}.slot")


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2TerminalSlice:
    terminal_ref: str
    assignment_ref: str
    byte_offset: int
    size_bytes: int
    shape: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.terminal_ref, f"{path}.terminal_ref")
        validate_nonempty(self.assignment_ref, f"{path}.assignment_ref")
        validate_uint64(self.byte_offset, f"{path}.byte_offset")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if not self.shape:
            raise SchemaError("terminal slice shape must be nonempty", path=f"{path}.shape")
        for index, extent in enumerate(self.shape):
            validate_uint64(extent, f"{path}.shape[{index}]")
            if extent == 0:
                raise SchemaError("terminal slice extent must be positive", path=f"{path}.shape[{index}]")

@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2Value:
    id: str
    rank: int
    origin_ref: str
    shape: tuple[int, ...]
    layout: str
    dtype: DType
    byte_offset: int
    size_bytes: int
    producer_task_ref: str | None
    consumer_task_refs: tuple[str, ...]
    buffer_ref: str | None
    terminal_ref: str | None
    borrowed: bool
    replicated: bool
    terminal_slices: tuple[MoeSwizzleIr2TerminalSlice, ...] = ()
    alias_source_refs: tuple[str, ...] = ()

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.origin_ref, f"{path}.origin_ref")
        if not self.shape:
            raise SchemaError("shape must be nonempty", path=f"{path}.shape")
        for index, extent in enumerate(self.shape):
            validate_uint64(extent, f"{path}.shape[{index}]")
            if extent == 0:
                raise SchemaError("shape extent must be positive", path=f"{path}.shape[{index}]")
        validate_nonempty(self.layout, f"{path}.layout")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        validate_uint64(self.byte_offset, f"{path}.byte_offset")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes != prod(self.shape) * _dtype_bytes(self.dtype):
            raise SchemaError("size must exactly equal typed shape extent", path=f"{path}.size_bytes")
        if self.producer_task_ref is not None:
            validate_nonempty(self.producer_task_ref, f"{path}.producer_task_ref")
        _unique_nonempty(self.consumer_task_refs, f"{path}.consumer_task_refs")
        if self.buffer_ref is not None:
            validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        if self.terminal_ref is not None:
            validate_nonempty(self.terminal_ref, f"{path}.terminal_ref")
        if type(self.terminal_slices) is not tuple:
            raise SchemaError("terminal slices must be an immutable tuple", path=f"{path}.terminal_slices")
        _unique_nonempty(self.alias_source_refs, f"{path}.alias_source_refs")
        for index, item in enumerate(self.terminal_slices):
            item.validate(f"{path}.terminal_slices[{index}]")
            if item.size_bytes != prod(item.shape) * _dtype_bytes(self.dtype):
                raise SchemaError("terminal slice bytes disagree with shape/dtype", path=f"{path}.terminal_slices[{index}]")
            if item.byte_offset + item.size_bytes > self.size_bytes:
                raise SchemaError("terminal slice exceeds its value", path=f"{path}.terminal_slices[{index}]")
        if self.terminal_ref is not None and self.terminal_slices:
            raise SchemaError("terminal value must use singular or sliced provenance", path=path)
        if (self.terminal_ref is not None or self.terminal_slices) and self.alias_source_refs:
            raise SchemaError("terminal value cannot be a composite alias", path=path)
        if self.terminal_slices:
            refs = tuple(item.terminal_ref for item in self.terminal_slices)
            assignments = tuple(item.assignment_ref for item in self.terminal_slices)
            if len(refs) != len(set(refs)) or len(assignments) != len(set(assignments)):
                raise SchemaError("terminal slices require unique terminal/assignment refs", path=f"{path}.terminal_slices")
            cursor = -1
            for item in sorted(self.terminal_slices, key=lambda part: part.byte_offset):
                if item.byte_offset < cursor:
                    raise SchemaError("terminal slices must not overlap", path=f"{path}.terminal_slices")
                cursor = item.byte_offset + item.size_bytes
        if type(self.borrowed) is not bool or type(self.replicated) is not bool:
            raise SchemaError("borrowed/replicated must be bool", path=path)
        if self.borrowed != (self.producer_task_ref is None and not self.alias_source_refs):
            raise SchemaError("BORROWED value must exactly lack a producer or alias sources", path=path)
        if self.alias_source_refs and self.producer_task_ref is not None:
            raise SchemaError("alias view cannot also have a direct producer", path=path)
        if len(self.alias_source_refs) > 1 and self.buffer_ref is None:
            raise SchemaError(
                "composite alias requires an explicit packed buffer",
                path=path,
            )
        if self.replicated and not self.borrowed:
            raise SchemaError("only BORROWED inputs may be replicated", path=f"{path}.replicated")
        if (self.terminal_ref is not None or self.terminal_slices) and (self.borrowed or self.replicated):
            raise SchemaError("terminal output must be an owned non-replicated value", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2Buffer:
    rank: int
    buffer_ref: str
    value_refs: tuple[str, ...]
    size_bytes: int
    slot_count: int

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        validate_nonempty(self.buffer_ref, f"{path}.buffer_ref")
        _unique_nonempty(self.value_refs, f"{path}.value_refs", allow_empty=False)
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.size_bytes")
        if self.slot_count not in (1, 2):
            raise SchemaError("MoE Swizzle supports exactly one or two slots", path=f"{path}.slot_count")


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2Task:
    id: str
    rank: int
    die_id: int
    kind: SwizzleActionKind
    work_role: str
    deps: tuple[str, ...]
    read_value_refs: tuple[str, ...]
    write_value_refs: tuple[str, ...]
    buffer_uses: tuple[MoeSwizzleIr2BufferUse, ...]
    assignment_refs: tuple[str, ...]
    expert_index: int | None
    tile_index: int | None
    n_block: int | None
    packet_ref: str | None
    stage: int | None
    pivot_rank: int | None
    original_action_refs: tuple[str, ...]
    pipeline_index: int
    buffer_slot: int | None
    buffer_family: str | None
    peer_rank: int | None
    flow_ref: str | None
    route_ref: str | None
    logical_bytes: int
    flops: int
    matmul_m: int | None
    matmul_n: int | None
    matmul_k: int | None
    dtype: DType | None
    accumulation_dtype: DType | None

    @property
    def work_key(self) -> tuple[object, ...]:
        return (
            -1 if self.expert_index is None else self.expert_index,
            -1 if self.tile_index is None else self.tile_index,
            -1 if self.n_block is None else self.n_block,
            self.assignment_refs,
        )

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_uint64(self.rank, f"{path}.rank")
        validate_uint64(self.die_id, f"{path}.die_id")
        if type(self.kind) is not SwizzleActionKind:
            raise SchemaError("must be a SwizzleActionKind", path=f"{path}.kind")
        validate_nonempty(self.work_role, f"{path}.work_role")
        for name in ("deps", "read_value_refs", "write_value_refs", "assignment_refs", "original_action_refs"):
            _unique_nonempty(getattr(self, name), f"{path}.{name}", allow_empty=name != "assignment_refs")
        for index, use in enumerate(self.buffer_uses):
            use.validate(f"{path}.buffer_uses[{index}]")
        if len({item.buffer_ref for item in self.buffer_uses}) != len(self.buffer_uses):
            raise SchemaError("task has duplicate buffer uses", path=f"{path}.buffer_uses")
        for name in ("expert_index", "tile_index", "n_block", "stage", "pivot_rank", "peer_rank"):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        for name in ("packet_ref", "flow_ref", "route_ref"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        validate_uint64(self.pipeline_index, f"{path}.pipeline_index")
        if (self.buffer_slot is None) != (self.buffer_family is None):
            raise SchemaError("buffer slot/family must be jointly present", path=path)
        if self.buffer_slot is not None:
            validate_uint64(self.buffer_slot, f"{path}.buffer_slot")
            validate_nonempty(self.buffer_family, f"{path}.buffer_family")
            if self.buffer_slot not in (0, 1) or self.buffer_family not in (
                "dispatch_operand", "combine_output",
            ):
                raise SchemaError("unsupported fixed buffer slot/family", path=path)
        validate_uint64(self.logical_bytes, f"{path}.logical_bytes")
        validate_uint64(self.flops, f"{path}.flops")
        if self.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            if self.peer_rank is None or self.flow_ref is None or self.packet_ref is None or self.stage is None or self.route_ref is None or self.logical_bytes == 0 or self.flops:
                raise SchemaError("transport requires peer/flow/packet/stage provenance", path=path)
        elif self.kind is SwizzleActionKind.WAIT:
            if self.peer_rank is not None or self.flow_ref is None or self.packet_ref is None or self.stage is None:
                raise SchemaError("WAIT requires flow/packet/stage and no peer rank", path=path)
        elif self.peer_rank is not None or self.flow_ref is not None or self.route_ref is not None:
            raise SchemaError("non-transport task cannot carry peer/flow/route", path=path)
        if self.kind is SwizzleActionKind.WAIT and len(self.deps) != 1:
            raise SchemaError("WAIT must depend on one RECV", path=f"{path}.deps")
        if self.kind is SwizzleActionKind.REDUCE and (
            len(self.read_value_refs), len(self.write_value_refs)
        ) != (2, 1):
            raise SchemaError("MoE reduction must be exactly binary", path=path)
        if self.kind is SwizzleActionKind.LOCAL_COPY and (
            len(self.read_value_refs), len(self.write_value_refs)
        ) != (1, 1):
            raise SchemaError("LOCAL_COPY must have one source and destination", path=path)
        if self.kind is SwizzleActionKind.SWIGLU and (
            (len(self.read_value_refs), len(self.write_value_refs)) != (1, 1)
            or self.logical_bytes == 0 or self.flops
        ):
            raise SchemaError("SWIGLU must have one flat input/output byte contract", path=path)

        matmul = (self.matmul_m, self.matmul_n, self.matmul_k)
        if self.kind is SwizzleActionKind.COMP:
            if any(item is None or item == 0 for item in matmul) or type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType or self.logical_bytes or self.flops != 2 * self.matmul_m * self.matmul_n * self.matmul_k or (len(self.read_value_refs), len(self.write_value_refs)) != (2, 1):
                raise SchemaError("COMP requires an exact typed MATMUL contract", path=path)
        elif any(item is not None for item in matmul) or self.dtype is not None or self.accumulation_dtype is not None:
            raise SchemaError("non-COMP task cannot carry MATMUL metadata", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2PacketSlice:
    assignment_ref: str
    source_offset_bytes: int
    destination_offset_bytes: int
    size_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.assignment_ref, f"{path}.assignment_ref")
        for name in ("source_offset_bytes", "destination_offset_bytes", "size_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.size_bytes == 0:
            raise SchemaError("packet slice must be nonempty", path=f"{path}.size_bytes")


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2Flow:
    id: str
    packet_ref: str
    stage: int
    pivot_rank: int | None
    source_rank: int
    destination_rank: int
    source_die_id: int
    destination_die_id: int
    route_ref: str
    die_path: tuple[int, ...]
    logical_bytes: int
    assignment_slices: tuple[MoeSwizzleIr2PacketSlice, ...]
    send_task_ref: str
    recv_task_ref: str
    wait_task_ref: str

    def validate(self, path: str) -> None:
        for name in ("id", "packet_ref", "route_ref", "send_task_ref", "recv_task_ref", "wait_task_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in ("stage", "source_rank", "destination_rank", "source_die_id", "destination_die_id", "logical_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.pivot_rank is not None:
            validate_uint64(self.pivot_rank, f"{path}.pivot_rank")
        if self.source_rank == self.destination_rank or self.source_die_id == self.destination_die_id:
            raise SchemaError("flow endpoints must be remote", path=path)
        if len(self.die_path) < 2 or self.die_path[0] != self.source_die_id or self.die_path[-1] != self.destination_die_id:
            raise SchemaError("die path endpoints are not exact", path=f"{path}.die_path")
        for index, die_id in enumerate(self.die_path):
            validate_uint64(die_id, f"{path}.die_path[{index}]")
        if not self.assignment_slices:
            raise SchemaError("flow requires assignment slices", path=f"{path}.assignment_slices")
        for index, item in enumerate(self.assignment_slices):
            item.validate(f"{path}.assignment_slices[{index}]")
        if len({item.assignment_ref for item in self.assignment_slices}) != len(self.assignment_slices):
            raise SchemaError("flow duplicates an assignment slice", path=f"{path}.assignment_slices")
        if self.logical_bytes == 0 or self.logical_bytes != sum(item.size_bytes for item in self.assignment_slices):
            raise SchemaError("flow bytes do not equal assignment slices", path=f"{path}.logical_bytes")


@dataclass(frozen=True, slots=True)
class MoeSwizzleIr2Projection:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    source_overlay_id: str
    tasks: tuple[MoeSwizzleIr2Task, ...]
    values: tuple[MoeSwizzleIr2Value, ...]
    buffers: tuple[MoeSwizzleIr2Buffer, ...]
    flows: tuple[MoeSwizzleIr2Flow, ...]
    terminal_refs: tuple[str, ...]
    endpoint_session_capacity: int

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleIr2Projection":
        result = cls(
            schema_version=MOE_SWIZZLE_IR2_SCHEMA_VERSION,
            producer_pass="project_moe_swizzle_ir2",
            id=stable_artifact_id(
                "moe_swizzle_ir2_projection",
                semantic,
                schema_version=MOE_SWIZZLE_IR2_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_execution_id", "source_overlay_id", "tasks", "values",
                "buffers", "flows", "terminal_refs", "endpoint_session_capacity",
            )
        }

    def validate(self, path: str = "moe_swizzle_ir2") -> None:
        if self.schema_version != MOE_SWIZZLE_IR2_SCHEMA_VERSION or self.producer_pass != "project_moe_swizzle_ir2":
            raise SchemaError("unsupported MoE Swizzle IR2 schema/producer", path=path)
        validate_nonempty(self.source_execution_id, f"{path}.source_execution_id")
        validate_nonempty(self.source_overlay_id, f"{path}.source_overlay_id")
        validate_uint64(self.endpoint_session_capacity, f"{path}.endpoint_session_capacity")
        if self.endpoint_session_capacity == 0:
            raise SchemaError("endpoint capacity must be positive", path=f"{path}.endpoint_session_capacity")
        _unique_nonempty(self.terminal_refs, f"{path}.terminal_refs", allow_empty=False)
        tasks = {}
        for index, task in enumerate(self.tasks):
            task.validate(f"{path}.tasks[{index}]")
            if task.id in tasks:
                raise SchemaError("duplicate task id", path=f"{path}.tasks[{index}]")
            tasks[task.id] = task
        values = {}
        for index, value in enumerate(self.values):
            value.validate(f"{path}.values[{index}]")
            if value.id in values:
                raise SchemaError("duplicate value id", path=f"{path}.values[{index}]")
            values[value.id] = value
        buffers = {}
        for index, buffer in enumerate(self.buffers):
            buffer.validate(f"{path}.buffers[{index}]")
            key = (buffer.rank, buffer.buffer_ref)
            if key in buffers:
                raise SchemaError("duplicate rank buffer", path=f"{path}.buffers[{index}]")
            buffers[key] = buffer
        if any(dep not in tasks for task in tasks.values() for dep in task.deps):
            raise SchemaError("task dependency is unknown", path=f"{path}.tasks")
        if any(ref not in values for task in tasks.values() for ref in task.read_value_refs + task.write_value_refs):
            raise SchemaError("task operand value is unknown", path=f"{path}.tasks")
        for value in values.values():
            if value.producer_task_ref is not None:
                producer = tasks.get(value.producer_task_ref)
                if producer is None or value.id not in producer.write_value_refs or producer.rank != value.rank:
                    raise SchemaError("value producer closure mismatch", path=f"{path}.values")
            if tuple(sorted(value.consumer_task_refs)) != tuple(sorted(task.id for task in tasks.values() if value.id in task.read_value_refs)):
                raise SchemaError("value consumer closure mismatch", path=f"{path}.values")
            if any(tasks[ref].rank != value.rank for ref in value.consumer_task_refs):
                raise SchemaError("value cannot cross ranks without SEND/RECV values", path=f"{path}.values")
            if value.buffer_ref is not None:
                buffer = buffers.get((value.rank, value.buffer_ref))
                if buffer is None or value.id not in buffer.value_refs or value.byte_offset + value.size_bytes > buffer.size_bytes:
                    raise SchemaError("buffered value exceeds or escapes its buffer", path=f"{path}.values")
            if value.alias_source_refs:
                sources = [values.get(ref) for ref in value.alias_source_refs]
                if len(sources) == 1 and value.buffer_ref is None:
                    source = sources[0]
                    if (
                        source is None
                        or source.alias_source_refs
                        or source.rank != value.rank
                        or value.byte_offset + value.size_bytes > source.size_bytes
                    ):
                        raise SchemaError(
                            "single-source alias is not a contained storage subview",
                            path=f"{path}.values",
                        )
                    continue
                if any(source is None or source.alias_source_refs or source.rank != value.rank or source.buffer_ref != value.buffer_ref for source in sources):
                    raise SchemaError("composite alias sources do not share one physical buffer", path=f"{path}.values")
                intervals = sorted((source.byte_offset, source.byte_offset + source.size_bytes) for source in sources)
                cursor = value.byte_offset
                for start, end in intervals:
                    if start != cursor:
                        raise SchemaError("composite alias sources must exactly and contiguously cover the view", path=f"{path}.values")
                    cursor = end
                if cursor != value.byte_offset + value.size_bytes:
                    raise SchemaError("composite alias source extent is not exact", path=f"{path}.values")
        for task in tasks.values():
            if task.kind is not SwizzleActionKind.SWIGLU:
                continue
            input_value = values[task.read_value_refs[0]]
            output_value = values[task.write_value_refs[0]]
            if (
                input_value.dtype is not DType.FP16
                or output_value.dtype is not DType.FP16
                or input_value.size_bytes != 2 * task.logical_bytes
                or output_value.size_bytes != task.logical_bytes
                or input_value.shape != (2 * output_value.shape[0], output_value.shape[1])
            ):
                raise SchemaError("SWIGLU flat input/output shape closure drifted", path=f"{path}.tasks")
        for buffer in buffers.values():
            if set(buffer.value_refs) != {value.id for value in values.values() if value.rank == buffer.rank and value.buffer_ref == buffer.buffer_ref}:
                raise SchemaError("buffer value coverage is not exact", path=f"{path}.buffers")
        for task in tasks.values():
            use_by_buffer = {item.buffer_ref: item.slot for item in task.buffer_uses}
            referenced = {
                values[ref].buffer_ref
                for ref in task.read_value_refs + task.write_value_refs
                if values[ref].buffer_ref is not None
            }
            if set(use_by_buffer) != referenced:
                raise SchemaError("task buffer uses do not exactly cover buffered operands", path=f"{path}.tasks")
            for buffer_ref, slot in use_by_buffer.items():
                buffer = buffers[(task.rank, buffer_ref)]
                if slot >= buffer.slot_count or (buffer.slot_count == 2 and slot != task.pipeline_index % 2):
                    raise SchemaError("buffer slot is not the exact fixed pipeline slot", path=f"{path}.tasks")
        flow_ids = set()
        flow_tasks = set()
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            if flow.id in flow_ids:
                raise SchemaError("duplicate flow id", path=f"{path}.flows[{index}]")
            flow_ids.add(flow.id)
            send, recv, wait = (tasks.get(ref) for ref in (flow.send_task_ref, flow.recv_task_ref, flow.wait_task_ref))
            if send is None or recv is None or wait is None or (send.kind, recv.kind, wait.kind) != (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT):
                raise SchemaError("flow task kinds are not exact", path=f"{path}.flows[{index}]")
            if (
                (send.rank, recv.rank, wait.rank) != (flow.source_rank, flow.destination_rank, flow.destination_rank)
                or (send.die_id, recv.die_id, wait.die_id) != (flow.source_die_id, flow.destination_die_id, flow.destination_die_id)
                or wait.deps != (recv.id,)
            ):
                raise SchemaError("flow endpoint/wait closure mismatch", path=f"{path}.flows[{index}]")
            assignments = tuple(item.assignment_ref for item in flow.assignment_slices)
            if any(
                task.flow_ref != flow.id
                or task.packet_ref != flow.packet_ref
                or task.stage != flow.stage
                or task.pivot_rank != flow.pivot_rank
                or task.assignment_refs != assignments
                for task in (send, recv, wait)
            ) or send.route_ref != flow.route_ref or recv.route_ref != flow.route_ref or wait.route_ref is not None or (send.logical_bytes, recv.logical_bytes, wait.logical_bytes) != (flow.logical_bytes, flow.logical_bytes, 0):
                raise SchemaError("flow provenance mismatch", path=f"{path}.flows[{index}]")
            if send.peer_rank != flow.destination_rank or recv.peer_rank != flow.source_rank:
                raise SchemaError("flow peer ranks mismatch", path=f"{path}.flows[{index}]")
            flow_tasks.update((send.id, recv.id, wait.id))
        expected_flow_tasks = {task.id for task in tasks.values() if task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV, SwizzleActionKind.WAIT)}
        if flow_tasks != expected_flow_tasks:
            raise SchemaError("flows must cover every SEND/RECV/WAIT exactly", path=f"{path}.flows")
        actual_terminals = tuple(sorted(
            ref for value in values.values()
            for ref in ((value.terminal_ref,) if value.terminal_ref is not None else tuple(
                item.terminal_ref for item in value.terminal_slices
            ))
        ))
        if tuple(sorted(self.terminal_refs)) != actual_terminals:
            raise SchemaError("terminal refs are not exact", path=f"{path}.terminal_refs")
        # Kahn closure catches cycles without imposing one global stream order.
        remaining = {task.id: set(task.deps) for task in tasks.values()}
        ready = sorted(ref for ref, deps in remaining.items() if not deps)
        visited = []
        while ready:
            ref = ready.pop(0)
            visited.append(ref)
            for other in sorted(remaining):
                if ref in remaining[other]:
                    remaining[other].remove(ref)
                    if not remaining[other] and other not in visited and other not in ready:
                        ready.append(other)
                        ready.sort()
        if len(visited) != len(tasks):
            raise SchemaError("task DAG contains a cycle", path=f"{path}.tasks")
        expected = stable_artifact_id("moe_swizzle_ir2_projection", self._semantic(), schema_version=MOE_SWIZZLE_IR2_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError("unstable projection id", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("MoeSwizzle") or name.startswith("MOE_SWIZZLE")]
