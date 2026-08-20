"""Projection, schedule, and global carriers for fixed EP4 S3-Lite MoE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import ExperimentSpec
from .ir0 import GemmWorkload, SwiGluWorkload
from .lite_moe_dp4 import LiteMoeDp4IR0Adapter, LiteMoeDp4N4IR1, LiteMoeDp4PlacedIR1
from .n4 import FusionPartitionContext, InterDiePlanningContext
from .placement import PlacementContext


LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_projection/v1alpha1"
)
LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_schedule/v1alpha1"
)
LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_global/v1alpha1"
)
LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_moe_dp4_execution_case/v1alpha1"
)


class LiteMoeDp4TaskKind(str, Enum):
    DMA_IN = "dma_in"
    GEMM = "gemm"
    SWIGLU = "swiglu"
    SEND = "send"
    RECV = "recv"
    WAIT = "wait"


class LiteMoeDp4BufferAccess(str, Enum):
    READ = "read"
    WRITE = "write"


def _semantic(instance: object) -> dict[str, object]:
    return {
        name: getattr(instance, name)
        for name in instance.__dataclass_fields__
        if name not in ("schema_version", "producer_pass", "id")
    }


def _stable(instance: object, prefix: str, version: str, path: str) -> None:
    expected = stable_artifact_id(prefix, _semantic(instance), schema_version=version)
    if getattr(instance, "id") != expected:
        raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


def _tuple(value: object, path: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    return value


@dataclass(frozen=True, slots=True)
class LiteMoeDp4PackedSlice:
    logical_value_ref: str
    root_value_ref: str
    offset_bytes: int
    size_bytes: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.logical_value_ref, f"{path}.logical_value_ref")
        validate_nonempty(self.root_value_ref, f"{path}.root_value_ref")
        for name in ("offset_bytes", "size_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.logical_value_ref == self.root_value_ref or self.size_bytes != 64:
            raise SchemaError("requires exact 64B logical slice of packed gate/up root", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Task:
    id: str
    kind: LiteMoeDp4TaskKind
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
    workload: GemmWorkload | SwiGluWorkload | None
    packed_output: LiteMoeDp4PackedSlice | None

    def validate(self, path: str = "lite_moe_dp4_task") -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.kind) is not LiteMoeDp4TaskKind:
            raise SchemaError("must be a DP4 task kind", path=f"{path}.kind")
        if self.die_id not in (0, 1, 2, 3):
            raise SchemaError("task die must be 0..3", path=f"{path}.die_id")
        if type(self.dtype) is not DType:
            raise SchemaError("must use a typed dtype", path=f"{path}.dtype")
        for name in ("read_values", "write_values", "deps"):
            values = _tuple(getattr(self, name), f"{path}.{name}")
            if len(values) != len(set(values)):
                raise SchemaError("contains duplicates", path=f"{path}.{name}")
        transport = self.kind in (
            LiteMoeDp4TaskKind.SEND,
            LiteMoeDp4TaskKind.RECV,
            LiteMoeDp4TaskKind.WAIT,
        )
        if self.kind is LiteMoeDp4TaskKind.DMA_IN:
            if (
                not self.access_ref
                or not self.state_ref
                or not self.hbm_binding_ref
                or self.p2p_binding_ref is not None
                or self.pair_route_ref is not None
                or self.peer_die_id is not None
                or self.read_values
                or len(self.write_values) != 1
                or self.deps
                or self.workload is not None
                or self.packed_output is not None
                or self.bytes != 1024
                or self.dtype is not DType.FP16
            ):
                raise SchemaError("invalid exact DP4 DMA_IN", path=path)
        elif self.kind in (LiteMoeDp4TaskKind.GEMM, LiteMoeDp4TaskKind.SWIGLU):
            expected = GemmWorkload if self.kind is LiteMoeDp4TaskKind.GEMM else SwiGluWorkload
            if (
                any(
                    item is not None
                    for item in (
                        self.access_ref,
                        self.state_ref,
                        self.hbm_binding_ref,
                        self.p2p_binding_ref,
                        self.pair_route_ref,
                        self.peer_die_id,
                    )
                )
                or type(self.workload) is not expected
                or self.bytes != 0
                or self.dtype is not DType.FP16
                or len(self.write_values) != 1
            ):
                raise SchemaError("invalid exact DP4 compute task", path=path)
            self.workload.validate(f"{path}.workload")
            if self.kind is LiteMoeDp4TaskKind.GEMM:
                if len(self.read_values) != 2:
                    raise SchemaError("GEMM requires activation+weight", path=path)
            elif len(self.read_values) != 1:
                raise SchemaError("SwiGLU requires one packed root", path=path)
            if self.packed_output is not None:
                self.packed_output.validate(f"{path}.packed_output")
        elif transport:
            if (
                not self.p2p_binding_ref
                or not self.pair_route_ref
                or self.peer_die_id not in (0, 1, 2, 3)
                or self.peer_die_id == self.die_id
                or self.bytes != 32
                or self.dtype is not DType.FP16
                or any(
                    item is not None
                    for item in (
                        self.access_ref,
                        self.state_ref,
                        self.hbm_binding_ref,
                        self.workload,
                        self.packed_output,
                    )
                )
            ):
                raise SchemaError("invalid exact DP4 transport task", path=path)
            if self.kind is LiteMoeDp4TaskKind.SEND and (
                len(self.read_values) != 1 or self.write_values
            ):
                raise SchemaError("SEND operands are not exact", path=path)
            if self.kind is LiteMoeDp4TaskKind.RECV and (
                self.read_values or len(self.write_values) != 1 or self.deps
            ):
                raise SchemaError("RECV operands are not exact", path=path)
            if self.kind is LiteMoeDp4TaskKind.WAIT and (
                self.read_values or self.write_values or len(self.deps) != 1
            ):
                raise SchemaError("WAIT operands are not exact", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Flow:
    id: str
    p2p_binding_ref: str
    pair_route_ref: str
    token_index: int
    expert_index: int
    source_die_id: int
    destination_die_id: int
    source_value_ref: str
    destination_value_ref: str
    bytes: int
    dtype: DType
    send_task_ref: str
    recv_task_ref: str
    wait_task_ref: str

    def validate(self, path: str = "lite_moe_dp4_flow") -> None:
        for name in (
            "id",
            "p2p_binding_ref",
            "pair_route_ref",
            "source_value_ref",
            "destination_value_ref",
            "send_task_ref",
            "recv_task_ref",
            "wait_task_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            self.token_index not in (1, 2, 3, 4, 5, 6)
            or self.expert_index not in (0, 1, 2, 3)
            or self.source_die_id not in (0, 1, 2, 3)
            or self.destination_die_id not in (0, 1, 2, 3)
            or self.source_die_id == self.destination_die_id
            or self.bytes != 32
            or self.dtype is not DType.FP16
        ):
            raise SchemaError("invalid exact DP4 flow", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4ProjectedDie:
    die_id: int
    tasks: tuple[LiteMoeDp4Task, ...]

    def validate(self, path: str) -> None:
        expected = (20, 26, 26, 20)
        if self.die_id not in (0, 1, 2, 3) or len(self.tasks) != expected[self.die_id]:
            raise SchemaError("projected die task cardinality changed", path=path)
        for index, task in enumerate(self.tasks):
            if type(task) is not LiteMoeDp4Task:
                raise SchemaError("must be a DP4 task", path=f"{path}.tasks[{index}]")
            task.validate(f"{path}.tasks[{index}]")
            if task.die_id != self.die_id:
                raise SchemaError("task belongs to another die", path=f"{path}.tasks[{index}]")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Projection:
    schema_version: str
    producer_pass: str
    id: str
    source_n4_id: str
    source_ir1_id: str
    planning_context_id: str
    dies: tuple[LiteMoeDp4ProjectedDie, ...]
    flows: tuple[LiteMoeDp4Flow, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4Projection":
        result = cls(
            LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION,
            "lite_moe_dp4_projection",
            stable_artifact_id(
                "s3_lite_moe_dp4_projection",
                semantic,
                schema_version=LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_projection") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_projection"
        ):
            raise SchemaError("unsupported DP4 projection schema/producer", path=path)
        if tuple(item.die_id for item in self.dies) != (0, 1, 2, 3):
            raise SchemaError("dies must be canonical 0..3", path=f"{path}.dies")
        for index, die in enumerate(self.dies):
            die.validate(f"{path}.dies[{index}]")
        tasks = {task.id: task for die in self.dies for task in die.tasks}
        if len(tasks) != 92 or any(dep not in tasks for task in tasks.values() for dep in task.deps):
            raise SchemaError("requires 92 unique dependency-closed tasks", path=f"{path}.dies")
        if len(self.flows) != 12 or sum(item.bytes for item in self.flows) != 384:
            raise SchemaError("requires twelve 32B forward flows", path=f"{path}.flows")
        binding_refs = set()
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            if flow.p2p_binding_ref in binding_refs:
                raise SchemaError("duplicate flow binding", path=f"{path}.flows[{index}]")
            binding_refs.add(flow.p2p_binding_ref)
            send = tasks.get(flow.send_task_ref)
            recv = tasks.get(flow.recv_task_ref)
            wait = tasks.get(flow.wait_task_ref)
            if (
                send is None
                or recv is None
                or wait is None
                or (send.kind, recv.kind, wait.kind)
                != (
                    LiteMoeDp4TaskKind.SEND,
                    LiteMoeDp4TaskKind.RECV,
                    LiteMoeDp4TaskKind.WAIT,
                )
                or (send.die_id, recv.die_id, wait.die_id)
                != (flow.source_die_id, flow.destination_die_id, flow.destination_die_id)
                or wait.deps != (recv.id,)
            ):
                raise SchemaError("flow task closure mismatch", path=f"{path}.flows[{index}]")
        _stable(
            self,
            "s3_lite_moe_dp4_projection",
            LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4TaskPlacement:
    task_ref: str
    die_id: int
    core_ref: str
    ordinal: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_nonempty(self.core_ref, f"{path}.core_ref")
        if self.die_id not in (0, 1, 2, 3):
            raise SchemaError("placement die must be 0..3", path=f"{path}.die_id")
        validate_uint64(self.ordinal, f"{path}.ordinal")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4BufferBinding:
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
        if self.die_id not in (0, 1, 2, 3):
            raise SchemaError("buffer die must be 0..3", path=f"{path}.die_id")
        if (
            self.address % 64
            or self.size_bytes == 0
            or self.first_ordinal > self.last_ordinal
        ):
            raise SchemaError("invalid aligned buffer lifetime", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4Scheduled:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    source_n4_id: str
    placements: tuple[LiteMoeDp4TaskPlacement, ...]
    buffers: tuple[LiteMoeDp4BufferBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4Scheduled":
        result = cls(
            LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION,
            "lite_moe_dp4_schedule",
            stable_artifact_id(
                "s3_lite_moe_dp4_schedule",
                semantic,
                schema_version=LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_schedule") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_schedule"
        ):
            raise SchemaError("unsupported DP4 schedule schema/producer", path=path)
        if len(self.placements) != 92 or len({item.task_ref for item in self.placements}) != 92:
            raise SchemaError("must place exactly 92 unique tasks", path=f"{path}.placements")
        for index, item in enumerate(self.placements):
            item.validate(f"{path}.placements[{index}]")
        if tuple((item.die_id, item.ordinal) for item in self.placements) != tuple(
            sorted((item.die_id, item.ordinal) for item in self.placements)
        ):
            raise SchemaError("placements must use die/ordinal order", path=f"{path}.placements")
        if len(self.buffers) != 68:
            raise SchemaError("must bind exactly 68 production buffers", path=f"{path}.buffers")
        intervals = {}
        refs = set()
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
        _stable(
            self,
            "s3_lite_moe_dp4_schedule",
            LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4BufferUse:
    binding_ref: str
    access: LiteMoeDp4BufferAccess

    def validate(self, path: str) -> None:
        validate_nonempty(self.binding_ref, f"{path}.binding_ref")
        if type(self.access) is not LiteMoeDp4BufferAccess:
            raise SchemaError("invalid buffer access", path=f"{path}.access")


@dataclass(frozen=True, slots=True)
class LiteMoeDp4GlobalAction:
    id: str
    task_ref: str
    kind: LiteMoeDp4TaskKind
    die_id: int
    core_ref: str
    deps: tuple[str, ...]
    buffer_uses: tuple[LiteMoeDp4BufferUse, ...]
    flow_ref: str | None
    hbm_binding_ref: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.task_ref, f"{path}.task_ref")
        validate_nonempty(self.core_ref, f"{path}.core_ref")
        if type(self.kind) is not LiteMoeDp4TaskKind or self.die_id not in (0, 1, 2, 3):
            raise SchemaError("invalid DP4 action kind/die", path=path)
        for index, use in enumerate(self.buffer_uses):
            use.validate(f"{path}.buffer_uses[{index}]")
        if self.kind is LiteMoeDp4TaskKind.DMA_IN:
            validate_nonempty(self.hbm_binding_ref, f"{path}.hbm_binding_ref")
        elif self.kind in (
            LiteMoeDp4TaskKind.SEND,
            LiteMoeDp4TaskKind.RECV,
            LiteMoeDp4TaskKind.WAIT,
        ):
            validate_nonempty(self.flow_ref, f"{path}.flow_ref")
        elif self.flow_ref is not None or self.hbm_binding_ref is not None:
            raise SchemaError("compute action has external metadata", path=path)


@dataclass(frozen=True, slots=True)
class LiteMoeDp4GlobalDag:
    schema_version: str
    producer_pass: str
    id: str
    source_schedule_id: str
    source_projection_id: str
    source_n4_id: str
    actions: tuple[LiteMoeDp4GlobalAction, ...]
    combined_output_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4GlobalDag":
        result = cls(
            LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION,
            "lite_moe_dp4_global_action",
            stable_artifact_id(
                "s3_lite_moe_dp4_global",
                semantic,
                schema_version=LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_global") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_global_action"
        ):
            raise SchemaError("unsupported DP4 global schema/producer", path=path)
        if len(self.actions) != 92 or len(self.combined_output_refs) != 8:
            raise SchemaError("requires 92 actions and eight token outputs", path=path)
        ids = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            if action.id in ids:
                raise SchemaError("duplicate action id", path=f"{path}.actions[{index}]")
            ids.add(action.id)
        if any(dep not in ids for action in self.actions for dep in action.deps):
            raise SchemaError("action dependency is unknown", path=f"{path}.actions")
        if len(set(self.combined_output_refs)) != 8:
            raise SchemaError("combined outputs must be unique", path=f"{path}.combined_output_refs")
        _stable(
            self,
            "s3_lite_moe_dp4_global",
            LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION,
            path,
        )


@dataclass(frozen=True, slots=True)
class LiteMoeDp4ExecutionCase:
    schema_version: str
    producer_pass: str
    id: str
    experiment: ExperimentSpec
    adapter: LiteMoeDp4IR0Adapter
    placement_context: PlacementContext
    partition_context: FusionPartitionContext
    planning_context: InterDiePlanningContext
    placed: LiteMoeDp4PlacedIR1
    n4: LiteMoeDp4N4IR1
    projection: LiteMoeDp4Projection
    schedule: LiteMoeDp4Scheduled
    global_dag: LiteMoeDp4GlobalDag

    @classmethod
    def create(cls, **semantic: object) -> "LiteMoeDp4ExecutionCase":
        result = cls(
            LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION,
            "lite_moe_dp4_execution_case",
            stable_artifact_id(
                "s3_lite_moe_dp4_execution_case",
                semantic,
                schema_version=LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "lite_moe_dp4_execution_case") -> None:
        if (
            self.schema_version != LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION
            or self.producer_pass != "lite_moe_dp4_execution_case"
        ):
            raise SchemaError("unsupported DP4 execution case schema/producer", path=path)
        self.experiment.validate(f"{path}.experiment")
        self.adapter.validate(f"{path}.adapter")
        self.placement_context.validate(f"{path}.placement_context")
        self.partition_context.validate(f"{path}.partition_context")
        self.planning_context.validate(f"{path}.planning_context")
        self.placed.validate(f"{path}.placed")
        self.n4.validate(f"{path}.n4")
        self.projection.validate(f"{path}.projection")
        self.schedule.validate(f"{path}.schedule")
        self.global_dag.validate(f"{path}.global_dag")
        if (
            self.placed.source != self.adapter
            or self.n4.source != self.placed
            or self.projection.source_n4_id != self.n4.id
            or self.schedule.source_projection_id != self.projection.id
            or self.global_dag.source_schedule_id != self.schedule.id
        ):
            raise SchemaError("source-to-global provenance is not exact", path=path)
        _stable(
            self,
            "s3_lite_moe_dp4_execution_case",
            LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION,
            path,
        )


__all__ = [
    "LITE_MOE_DP4_EXECUTION_CASE_SCHEMA_VERSION",
    "LITE_MOE_DP4_GLOBAL_SCHEMA_VERSION",
    "LITE_MOE_DP4_PROJECTION_SCHEMA_VERSION",
    "LITE_MOE_DP4_SCHEDULE_SCHEMA_VERSION",
    "LiteMoeDp4BufferAccess",
    "LiteMoeDp4BufferBinding",
    "LiteMoeDp4BufferUse",
    "LiteMoeDp4ExecutionCase",
    "LiteMoeDp4Flow",
    "LiteMoeDp4GlobalAction",
    "LiteMoeDp4GlobalDag",
    "LiteMoeDp4PackedSlice",
    "LiteMoeDp4ProjectedDie",
    "LiteMoeDp4Projection",
    "LiteMoeDp4Scheduled",
    "LiteMoeDp4Task",
    "LiteMoeDp4TaskKind",
    "LiteMoeDp4TaskPlacement",
]
