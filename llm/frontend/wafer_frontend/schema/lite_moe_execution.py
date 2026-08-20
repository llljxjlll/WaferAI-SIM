"""Isolated projection, schedule, and global-action carriers for S3-Lite MoE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import GemmWorkload, SwiGluWorkload


LITE_MOE_PROJECTION_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_projection/v1alpha1"
)
LITE_MOE_SCHEDULE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_schedule/v1alpha1"
)
LITE_MOE_GLOBAL_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_global/v1alpha1"
)


class LiteMoeTaskKind(str, Enum):
    DMA_IN = "dma_in"
    GEMM = "gemm"
    SWIGLU = "swiglu"
    SEND = "send"
    RECV = "recv"
    WAIT = "wait"


class LiteMoeBufferAccess(str, Enum):
    READ = "read"
    WRITE = "write"


LiteMoeComputeWorkload = GemmWorkload | SwiGluWorkload


def _tuple(value: object, path: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    return value


@dataclass(frozen=True, slots=True)
class LiteMoePackedSlice:
    logical_value_ref: str
    root_value_ref: str
    offset_bytes: int
    size_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.logical_value_ref, f"{path}.logical_value_ref")
        validate_nonempty(self.root_value_ref, f"{path}.root_value_ref")
        validate_uint64(self.offset_bytes, f"{path}.offset_bytes")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.logical_value_ref == self.root_value_ref or self.size_bytes == 0:
            raise SchemaError("packed slice requires distinct logical/root values and non-zero size", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeTask:
    id: str
    kind: LiteMoeTaskKind
    die_id: int
    node_ref: str
    access_ref: str | None
    state_ref: str | None
    hbm_binding_ref: str | None
    p2p_binding_ref: str | None
    pair_route_ref: str | None
    peer_die_id: int | None
    bytes: int
    dtype: DType
    read_values: tuple[str, ...]
    write_values: tuple[str, ...]
    deps: tuple[str, ...]
    workload: LiteMoeComputeWorkload | None
    packed_output: LiteMoePackedSlice | None

    def validate(self, path: str = "lite_moe_task") -> None:
        validate_nonempty(self.id, f"{path}.id")
        if type(self.kind) is not LiteMoeTaskKind:
            raise SchemaError("must be a LiteMoeTaskKind", path=f"{path}.kind")
        validate_uint64(self.die_id, f"{path}.die_id")
        if self.die_id not in (0, 1):
            raise SchemaError("must be die 0 or 1", path=f"{path}.die_id")
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        validate_uint64(self.bytes, f"{path}.bytes")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        for field_name in ("read_values", "write_values", "deps"):
            values = _tuple(getattr(self, field_name), f"{path}.{field_name}")
            if len(values) != len(set(values)):
                raise SchemaError("contains duplicates", path=f"{path}.{field_name}")
            for index, value in enumerate(values):
                validate_nonempty(value, f"{path}.{field_name}[{index}]")

        if self.kind is LiteMoeTaskKind.DMA_IN:
            for field_name in ("access_ref", "state_ref", "hbm_binding_ref"):
                validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
            if (
                self.p2p_binding_ref is not None
                or self.pair_route_ref is not None
                or self.peer_die_id is not None
                or self.read_values
                or len(self.write_values) != 1
                or self.deps
                or self.workload is not None
                or self.packed_output is not None
                or self.bytes == 0
            ):
                raise SchemaError("invalid exact DMA_IN contract", path=path)
            return

        if self.kind in (LiteMoeTaskKind.GEMM, LiteMoeTaskKind.SWIGLU):
            if any(
                value is not None
                for value in (
                    self.access_ref,
                    self.state_ref,
                    self.hbm_binding_ref,
                    self.p2p_binding_ref,
                    self.pair_route_ref,
                    self.peer_die_id,
                )
            ):
                raise SchemaError("compute task contains non-compute metadata", path=path)
            expected = (
                GemmWorkload
                if self.kind is LiteMoeTaskKind.GEMM
                else SwiGluWorkload
            )
            if type(self.workload) is not expected:
                raise SchemaError("compute workload kind mismatch", path=f"{path}.workload")
            self.workload.validate(f"{path}.workload")
            expected_reads = 2 if self.kind is LiteMoeTaskKind.GEMM else 1
            if len(self.read_values) != expected_reads or len(self.write_values) != 1:
                raise SchemaError("invalid exact compute operand contract", path=path)
            if self.packed_output is not None:
                if self.kind is not LiteMoeTaskKind.GEMM:
                    raise SchemaError("only gate/up GEMM may write a packed slice", path=f"{path}.packed_output")
                self.packed_output.validate(f"{path}.packed_output")
                if self.write_values != (self.packed_output.root_value_ref,):
                    raise SchemaError("packed GEMM must write its root value", path=f"{path}.write_values")
            if self.bytes != 0:
                raise SchemaError("compute task bytes must be zero", path=f"{path}.bytes")
            return

        for field_name in ("p2p_binding_ref", "pair_route_ref"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if self.peer_die_id not in (0, 1) or self.peer_die_id == self.die_id:
            raise SchemaError("transport peer must be the other die", path=f"{path}.peer_die_id")
        if any(
            value is not None
            for value in (self.access_ref, self.state_ref, self.hbm_binding_ref)
        ) or self.workload is not None or self.packed_output is not None:
            raise SchemaError("transport task contains non-transport metadata", path=path)
        if self.bytes != 32 or self.dtype is not DType.FP16:
            raise SchemaError("S3-Lite transport must be exact FP16 32B", path=path)
        if self.kind is LiteMoeTaskKind.SEND:
            exact = (len(self.read_values), len(self.write_values), len(self.deps))
            if exact[0:2] != (1, 0):
                raise SchemaError("SEND must read one local value", path=path)
        elif self.kind is LiteMoeTaskKind.RECV:
            if self.read_values or len(self.write_values) != 1 or self.deps:
                raise SchemaError("RECV must write one local value", path=path)
        else:
            if self.read_values or self.write_values or len(self.deps) != 1:
                raise SchemaError("WAIT must depend on one RECV", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeFlow:
    id: str
    p2p_binding_ref: str
    pair_route_ref: str
    source_die_id: int
    destination_die_id: int
    source_value_ref: str
    destination_value_ref: str
    bytes: int
    dtype: DType
    send_task_ref: str
    recv_task_ref: str
    wait_task_ref: str

    def validate(self, path: str = "lite_moe_flow") -> None:
        for field_name in (
            "id", "p2p_binding_ref", "pair_route_ref", "source_value_ref",
            "destination_value_ref", "send_task_ref", "recv_task_ref",
            "wait_task_ref",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if (
            self.source_die_id not in (0, 1)
            or self.destination_die_id not in (0, 1)
            or self.source_die_id == self.destination_die_id
            or self.bytes != 32
            or self.dtype is not DType.FP16
        ):
            raise SchemaError("invalid exact S3-Lite flow geometry", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeProjectedDie:
    die_id: int
    tasks: tuple[LiteMoeTask, ...]

    def validate(self, path: str) -> None:
        if self.die_id not in (0, 1):
            raise SchemaError("must be die 0 or 1", path=f"{path}.die_id")
        _tuple(self.tasks, f"{path}.tasks")
        if len(self.tasks) != 40:
            raise SchemaError("must contain exactly 40 local tasks", path=f"{path}.tasks")
        for index, task in enumerate(self.tasks):
            if type(task) is not LiteMoeTask:
                raise SchemaError("must be a LiteMoeTask", path=f"{path}.tasks[{index}]")
            task.validate(f"{path}.tasks[{index}]")
            if task.die_id != self.die_id:
                raise SchemaError("task belongs to another die", path=f"{path}.tasks[{index}]")


@dataclass(frozen=True, slots=True)
class LiteMoeProjection:
    schema_version: str
    producer_pass: str
    id: str
    source_n4_id: str
    source_ir1_id: str
    planning_context_id: str
    dies: tuple[LiteMoeProjectedDie, ...]
    flows: tuple[LiteMoeFlow, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeProjection":
        result = cls(
            schema_version=LITE_MOE_PROJECTION_SCHEMA_VERSION,
            producer_pass="lite_moe_projection",
            id=stable_artifact_id(
                "s3_lite_static_moe_projection", semantic_key,
                schema_version=LITE_MOE_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in ("schema_version", "producer_pass", "id")}

    def validate(self, path: str = "lite_moe_projection") -> None:
        if self.schema_version != LITE_MOE_PROJECTION_SCHEMA_VERSION or self.producer_pass != "lite_moe_projection":
            raise SchemaError("unsupported projection schema/producer", path=path)
        for name in ("source_n4_id", "source_ir1_id", "planning_context_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if tuple(item.die_id for item in self.dies) != (0, 1):
            raise SchemaError("dies must be canonical (0, 1)", path=f"{path}.dies")
        for index, die in enumerate(self.dies):
            die.validate(f"{path}.dies[{index}]")
        task_ids = [task.id for die in self.dies for task in die.tasks]
        if len(task_ids) != 80 or len(task_ids) != len(set(task_ids)):
            raise SchemaError("must contain 80 unique tasks", path=f"{path}.dies")
        task_index = {task.id: task for die in self.dies for task in die.tasks}
        for task in task_index.values():
            if any(dep not in task_index for dep in task.deps):
                raise SchemaError("task dependency is unknown", path=f"{path}.dies")
        if len(self.flows) != 8:
            raise SchemaError("must contain exactly eight remote flows", path=f"{path}.flows")
        flow_ids: set[str] = set()
        binding_ids: set[str] = set()
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            if flow.id in flow_ids or flow.p2p_binding_ref in binding_ids:
                raise SchemaError("duplicate flow/binding identity", path=f"{path}.flows[{index}]")
            flow_ids.add(flow.id); binding_ids.add(flow.p2p_binding_ref)
            send = task_index.get(flow.send_task_ref)
            recv = task_index.get(flow.recv_task_ref)
            wait = task_index.get(flow.wait_task_ref)
            if (
                send is None or recv is None or wait is None
                or send.kind is not LiteMoeTaskKind.SEND
                or recv.kind is not LiteMoeTaskKind.RECV
                or wait.kind is not LiteMoeTaskKind.WAIT
                or (send.die_id, recv.die_id, wait.die_id)
                != (flow.source_die_id, flow.destination_die_id, flow.destination_die_id)
                or send.p2p_binding_ref != flow.p2p_binding_ref
                or recv.p2p_binding_ref != flow.p2p_binding_ref
                or wait.p2p_binding_ref != flow.p2p_binding_ref
                or send.read_values != (flow.source_value_ref,)
                or recv.write_values != (flow.destination_value_ref,)
                or wait.deps != (recv.id,)
            ):
                raise SchemaError("flow task closure mismatch", path=f"{path}.flows[{index}]")
        expected_id = stable_artifact_id("s3_lite_static_moe_projection", self._semantic_key(), schema_version=LITE_MOE_PROJECTION_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeTaskPlacement:
    task_ref: str
    die_id: int
    core_ref: str
    ordinal: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        if self.die_id not in (0, 1):
            raise SchemaError("must be die 0 or 1", path=f"{path}.die_id")
        validate_nonempty(self.core_ref, f"{path}.core_ref")
        validate_uint64(self.ordinal, f"{path}.ordinal")


@dataclass(frozen=True, slots=True)
class LiteMoeBufferBinding:
    id: str
    value_ref: str
    die_id: int
    core_ref: str
    address: int
    size_bytes: int
    first_ordinal: int
    last_ordinal: int

    def validate(self, path: str) -> None:
        for name in ("id", "value_ref", "core_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.die_id not in (0, 1):
            raise SchemaError("must be die 0 or 1", path=f"{path}.die_id")
        for name in ("address", "size_bytes", "first_ordinal", "last_ordinal"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.size_bytes == 0 or self.address % 64 or self.first_ordinal > self.last_ordinal:
            raise SchemaError("invalid aligned buffer lifetime", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeScheduled:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    source_n4_id: str
    placements: tuple[LiteMoeTaskPlacement, ...]
    buffers: tuple[LiteMoeBufferBinding, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeScheduled":
        result = cls(
            schema_version=LITE_MOE_SCHEDULE_SCHEMA_VERSION,
            producer_pass="lite_moe_schedule",
            id=stable_artifact_id("s3_lite_static_moe_schedule", semantic_key, schema_version=LITE_MOE_SCHEDULE_SCHEMA_VERSION),
            **semantic_key,
        )
        result.validate(); return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in ("schema_version", "producer_pass", "id")}

    def validate(self, path: str = "lite_moe_schedule") -> None:
        if self.schema_version != LITE_MOE_SCHEDULE_SCHEMA_VERSION or self.producer_pass != "lite_moe_schedule":
            raise SchemaError("unsupported schedule schema/producer", path=path)
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        validate_nonempty(self.source_n4_id, f"{path}.source_n4_id")
        if len(self.placements) != 80:
            raise SchemaError("must place exactly 80 tasks", path=f"{path}.placements")
        for index, item in enumerate(self.placements): item.validate(f"{path}.placements[{index}]")
        if len({item.task_ref for item in self.placements}) != 80:
            raise SchemaError("task placements must be unique", path=f"{path}.placements")
        if tuple((item.die_id, item.ordinal) for item in self.placements) != tuple(sorted((item.die_id, item.ordinal) for item in self.placements)):
            raise SchemaError("placements must use die/ordinal order", path=f"{path}.placements")
        intervals: dict[tuple[int, str], list[tuple[int, int]]] = {}
        refs: set[tuple[int, str]] = set()
        for index, item in enumerate(self.buffers):
            item.validate(f"{path}.buffers[{index}]")
            key = (item.die_id, item.value_ref)
            if key in refs:
                raise SchemaError("duplicate local value binding", path=f"{path}.buffers[{index}]")
            refs.add(key)
            ranges = intervals.setdefault((item.die_id, item.core_ref), [])
            if any(item.address < end and start < item.address + item.size_bytes for start, end in ranges):
                raise SchemaError("SRAM buffer overlap", path=f"{path}.buffers[{index}]")
            ranges.append((item.address, item.address + item.size_bytes))
        expected_id = stable_artifact_id("s3_lite_static_moe_schedule", self._semantic_key(), schema_version=LITE_MOE_SCHEDULE_SCHEMA_VERSION)
        if self.id != expected_id: raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class LiteMoeBufferUse:
    binding_ref: str
    access: LiteMoeBufferAccess

    def validate(self, path: str) -> None:
        validate_nonempty(self.binding_ref, f"{path}.binding_ref")
        if type(self.access) is not LiteMoeBufferAccess:
            raise SchemaError("must be a LiteMoeBufferAccess", path=f"{path}.access")


@dataclass(frozen=True, slots=True)
class LiteMoeGlobalAction:
    id: str
    task_ref: str
    kind: LiteMoeTaskKind
    die_id: int
    core_ref: str
    deps: tuple[str, ...]
    buffer_uses: tuple[LiteMoeBufferUse, ...]
    flow_ref: str | None
    hbm_binding_ref: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id"); validate_nonempty(self.task_ref, f"{path}.task_ref")
        if type(self.kind) is not LiteMoeTaskKind: raise SchemaError("invalid action kind", path=f"{path}.kind")
        if self.die_id not in (0, 1): raise SchemaError("must be die 0 or 1", path=f"{path}.die_id")
        validate_nonempty(self.core_ref, f"{path}.core_ref")
        _tuple(self.deps, f"{path}.deps"); _tuple(self.buffer_uses, f"{path}.buffer_uses")
        for index, use in enumerate(self.buffer_uses): use.validate(f"{path}.buffer_uses[{index}]")
        if self.kind is LiteMoeTaskKind.DMA_IN:
            validate_nonempty(self.hbm_binding_ref, f"{path}.hbm_binding_ref")
            if self.flow_ref is not None: raise SchemaError("DMA cannot reference flow", path=path)
        elif self.kind in (LiteMoeTaskKind.SEND, LiteMoeTaskKind.RECV, LiteMoeTaskKind.WAIT):
            validate_nonempty(self.flow_ref, f"{path}.flow_ref")
            if self.hbm_binding_ref is not None: raise SchemaError("transport cannot reference HBM", path=path)
        elif self.flow_ref is not None or self.hbm_binding_ref is not None:
            raise SchemaError("compute action contains external metadata", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeGlobalDag:
    schema_version: str
    producer_pass: str
    id: str
    source_schedule_id: str
    source_projection_id: str
    source_n4_id: str
    actions: tuple[LiteMoeGlobalAction, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeGlobalDag":
        result = cls(
            schema_version=LITE_MOE_GLOBAL_SCHEMA_VERSION,
            producer_pass="lite_moe_global_action",
            id=stable_artifact_id("s3_lite_static_moe_global", semantic_key, schema_version=LITE_MOE_GLOBAL_SCHEMA_VERSION),
            **semantic_key,
        )
        result.validate(); return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name not in ("schema_version", "producer_pass", "id")}

    def validate(self, path: str = "lite_moe_global") -> None:
        if self.schema_version != LITE_MOE_GLOBAL_SCHEMA_VERSION or self.producer_pass != "lite_moe_global_action":
            raise SchemaError("unsupported global schema/producer", path=path)
        for name in ("source_schedule_id", "source_projection_id", "source_n4_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if len(self.actions) != 80:
            raise SchemaError("must contain exactly 80 actions", path=f"{path}.actions")
        ids: set[str] = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.id in ids: raise SchemaError("duplicate action id", path=f"{path}.actions[{index}]")
            ids.add(action.id)
        if any(dep not in ids for action in self.actions for dep in action.deps):
            raise SchemaError("action dependency is unknown", path=f"{path}.actions")
        expected_id = stable_artifact_id("s3_lite_static_moe_global", self._semantic_key(), schema_version=LITE_MOE_GLOBAL_SCHEMA_VERSION)
        if self.id != expected_id: raise SchemaError("unstable artifact id", path=f"{path}.id")


__all__ = [name for name in tuple(globals()) if name.startswith("LiteMoe") or name.startswith("LITE_MOE_")]
