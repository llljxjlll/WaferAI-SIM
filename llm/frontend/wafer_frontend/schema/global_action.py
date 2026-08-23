"""Exact, lossless quotient of scheduled per-die semantic tasks."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .action import ComputeContract, ReductionContract, SyncContract
from .common import DType, stable_artifact_id, validate_dependency_dag, validate_nonempty, validate_uint64
from .ir0 import OpKind
from .ir1 import IR1
from .persistent_state import PersistentStateAccess
from .ir2 import (
    BufferAccess,
    DmaContract,
    BufferUseRole,
    FlowRouteBinding,
    IR2ProjectionResult,
    IntraDieScheduleSet,
    LogicalRuntimeBinding,
    NodeOrigin,
    RegionLowering,
    SemanticFlow,
    SemanticTask,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
    StateUseAccess,
    TensorSlice,
)
from .state_transfer import (
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
    StateTransferContract,
)


GLOBAL_ACTION_SCHEMA_VERSION = "wafer_frontend.global_action/v1alpha8"
GLOBAL_ACTION_DAG_SCHEMA_VERSION = "wafer_frontend.global_action_dag/v1alpha11"
STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY = 3


def state_transfer_wave_task_dependencies(
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Return exact remote-completion wave edges for segmented P2P SENDs.

    The runtime keeps a synchronous TX session resident until its remote ACK
    arrives. Core order alone therefore cannot bound resident sessions. For
    each physical destination endpoint core, units are split into contiguous
    source-core runs.  A run keeps at most ``capacity`` requests resident; a
    source transition waits for the previous run's final completion so an
    inactive source mailbox never accumulates a full window of event credits.
    """

    schedule_by_dag = {
        schedule.dag_id: schedule for schedule in schedule_set.schedules
    }
    wait_by_unit: dict[
        tuple[str, int], tuple[int, int, int, str, str]
    ] = {}
    send_by_unit: dict[
        tuple[str, int], tuple[str, str, tuple[int, int]]
    ] = {}
    for dag in projection.dags:
        schedule = schedule_by_dag[dag.id]
        placement = {
            item.task_id: item.core_id for item in schedule.placements
        }
        position = {
            task_id: index
            for order in schedule.core_orders
            for index, task_id in enumerate(order.task_ids)
        }
        for task in dag.tasks:
            origin = task.origin_ref
            if (
                not isinstance(origin, StateTransferOrigin)
                or origin.segment_index is None
            ):
                continue
            unit = (origin.state_transfer_ref, origin.segment_index)
            if task.kind is SemanticTaskKind.WAIT:
                if unit in wait_by_unit:
                    raise SchemaError(
                        "segmented transfer unit has multiple WAIT completions",
                        path="projection.dags.tasks",
                    )
                wait_by_unit[unit] = (
                    dag.die_id,
                    placement[task.id],
                    position[task.id],
                    dag.id,
                    task.id,
                )
            elif task.kind is SemanticTaskKind.SEND:
                if unit in send_by_unit:
                    raise SchemaError(
                        "segmented transfer unit has multiple SEND actions",
                        path="projection.dags.tasks",
                    )
                send_by_unit[unit] = (
                    dag.id,
                    task.id,
                    (dag.die_id, placement[task.id]),
                )
    result: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    if set(send_by_unit) != set(wait_by_unit):
        raise SchemaError(
            "segmented transfer units require exactly one SEND and one WAIT",
            path="projection.dags.tasks",
        )
    units_by_destination_core: dict[
        tuple[int, int],
        list[
            tuple[
                int, str, int, str, str, str, str, tuple[int, int]
            ]
        ],
    ] = {}
    for unit, (send_dag, send_task, source_core) in send_by_unit.items():
        wait_die, wait_core, wait_position, wait_dag, wait_task = (
            wait_by_unit[unit]
        )
        transfer_ref, segment_index = unit
        units_by_destination_core.setdefault((wait_die, wait_core), []).append(
            (
                wait_position,
                transfer_ref,
                segment_index,
                wait_dag,
                wait_task,
                send_dag,
                send_task,
                source_core,
            )
        )
    for units in units_by_destination_core.values():
        units.sort()
        runs: list[list[tuple[object, ...]]] = []
        for unit in units:
            if not runs or runs[-1][-1][-1] != unit[-1]:
                runs.append([])
            runs[-1].append(unit)
        previous_run: list[tuple[object, ...]] | None = None
        for run in runs:
            if previous_run is not None:
                send_dag, send_task = run[0][5:7]
                prior_wait_dag, prior_wait_task = previous_run[-1][3:5]
                result[(send_dag, send_task)] = (
                    (prior_wait_dag, prior_wait_task),
                )
            for index in range(
                STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY, len(run)
            ):
                send_dag, send_task = run[index][5:7]
                prior_wait_dag, prior_wait_task = run[
                    index - STATE_TRANSFER_ENDPOINT_SESSION_CAPACITY
                ][3:5]
                result[(send_dag, send_task)] = (
                    (prior_wait_dag, prior_wait_task),
                )
            previous_run = run
    return result


@dataclass(frozen=True, slots=True)
class LogicalCoreRef:
    """Canonical frontend core identity; never a backend runtime core id."""

    die_id: int
    local_core_id: int

    def validate(self, path: str) -> None:
        validate_uint64(self.die_id, f"{path}.die_id")
        validate_uint64(self.local_core_id, f"{path}.local_core_id")


@dataclass(frozen=True, slots=True)
class ScheduledDagRef:
    dag_id: str
    schedule_id: str
    die_id: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.dag_id, f"{path}.dag_id")
        validate_nonempty(self.schedule_id, f"{path}.schedule_id")
        validate_uint64(self.die_id, f"{path}.die_id")


@dataclass(frozen=True, slots=True)
class ScheduledSourceRef:
    dag_id: str
    schedule_id: str
    task_id: str

    def validate(self, path: str) -> None:
        validate_nonempty(self.dag_id, f"{path}.dag_id")
        validate_nonempty(self.schedule_id, f"{path}.schedule_id")
        validate_nonempty(self.task_id, f"{path}.task_id")


@dataclass(frozen=True, slots=True)
class ActionBufferUse:
    binding_id: str
    access: BufferAccess
    role: BufferUseRole
    operand_index: int
    contribution_rank: int | None
    tensor_slice: TensorSlice

    def validate(self, path: str) -> None:
        validate_nonempty(self.binding_id, f"{path}.binding_id")
        if type(self.access) is not BufferAccess:
            raise SchemaError("must be a BufferAccess", path=f"{path}.access")
        if type(self.role) is not BufferUseRole:
            raise SchemaError("must be a BufferUseRole", path=f"{path}.role")
        validate_uint64(self.operand_index, f"{path}.operand_index")
        if type(self.tensor_slice) is not TensorSlice:
            raise SchemaError("must be a TensorSlice", path=f"{path}.tensor_slice")
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.contribution_rank is not None:
            validate_uint64(self.contribution_rank, f"{path}.contribution_rank")
        if self.role is BufferUseRole.REDUCE_INPUT:
            if self.contribution_rank is None:
                raise SchemaError("REDUCE_INPUT requires contribution_rank", path=f"{path}.contribution_rank")
        elif self.contribution_rank is not None:
            raise SchemaError("only REDUCE_INPUT may carry contribution_rank", path=f"{path}.contribution_rank")


@dataclass(frozen=True, slots=True)
class ActionStateUse:
    """One GlobalAction HBM endpoint, kept separate from SRAM BufferABI."""

    hbm_binding_ref: str
    access: StateUseAccess

    def validate(self, path: str) -> None:
        validate_nonempty(self.hbm_binding_ref, f"{path}.hbm_binding_ref")
        if type(self.access) is not StateUseAccess:
            raise SchemaError(
                "must be a StateUseAccess", path=f"{path}.access"
            )


@dataclass(frozen=True, slots=True)
class GlobalAction:
    """One source SemanticTask plus every lowering-relevant scheduled binding."""

    schema_version: str
    id: str
    source: ScheduledSourceRef
    task_kind: SemanticTaskKind
    origin_ref: NodeOrigin
    lowering: RegionLowering
    region_id: str
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
    dma: DmaContract | None
    logical_core: LogicalCoreRef | None
    core_order_index: int | None
    flow: SemanticFlow | None
    flow_route: FlowRouteBinding | None
    runtime_binding: LogicalRuntimeBinding | None
    buffer_uses: tuple[ActionBufferUse, ...]
    state_uses: tuple[ActionStateUse, ...]
    deps: tuple[str, ...]

    @staticmethod
    def stable_id_for_source(source: ScheduledSourceRef) -> str:
        return stable_artifact_id(
            "global_action",
            source,
            schema_version=GLOBAL_ACTION_SCHEMA_VERSION,
        )

    @classmethod
    def create(cls, **semantic_key: object) -> "GlobalAction":
        source = semantic_key.get("source")
        if not isinstance(source, ScheduledSourceRef):
            raise SchemaError("source must be a ScheduledSourceRef", path="global_action.source")
        return cls(
            schema_version=GLOBAL_ACTION_SCHEMA_VERSION,
            id=cls.stable_id_for_source(source),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        if self.schema_version != GLOBAL_ACTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.source.validate(f"{path}.source")
        expected_id = self.stable_id_for_source(self.source)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")
        validate_nonempty(self.region_id, f"{path}.region_id")
        SemanticTask(
            id=self.id,
            kind=self.task_kind,
            origin_ref=self.origin_ref,
            region_id=self.region_id,
            op_kind=self.op_kind,
            member_id=self.member_id,
            flow_id=self.flow_id,
            chunk_id=self.chunk_id,
            collective_step=self.collective_step,
            source_rank=self.source_rank,
            destination_rank=self.destination_rank,
            tensor_slice=self.tensor_slice,
            bytes=self.bytes,
            dtype=self.dtype,
            shape=self.shape,
            read_values=self.read_values,
            write_values=self.write_values,
            compute=self.compute,
            reduction=self.reduction,
            sync=self.sync,
            deps=(),
            dma=self.dma,
        ).validate(path)
        if self.lowering is RegionLowering.JSON_COARSE and not (
            (self.task_kind is SemanticTaskKind.COMP and self.compute is not None)
            or self.task_kind
            in (
                SemanticTaskKind.LOCAL_SEND,
                SemanticTaskKind.LOCAL_RECV,
                SemanticTaskKind.LOCAL_WAIT,
                SemanticTaskKind.LOCAL_COPY,
                SemanticTaskKind.REDUCE,
            )
        ):
            raise SchemaError("JSON_COARSE requires COMP, local transport, LOCAL_COPY, or REDUCE", path=path)
        if self.task_kind is SemanticTaskKind.TRANSIT:
            if self.logical_core is not None or self.core_order_index is not None:
                raise SchemaError("TRANSIT cannot carry a logical core", path=path)
            if self.buffer_uses or self.state_uses or self.runtime_binding is not None:
                raise SchemaError("TRANSIT cannot carry buffer/state/runtime bindings", path=path)
        else:
            if self.logical_core is None or self.core_order_index is None:
                raise SchemaError("executable action requires a logical core and order index", path=path)
            self.logical_core.validate(f"{path}.logical_core")
            validate_uint64(self.core_order_index, f"{path}.core_order_index")
        is_local_flow = (
            self.task_kind
            in (
                SemanticTaskKind.LOCAL_SEND,
                SemanticTaskKind.LOCAL_RECV,
                SemanticTaskKind.LOCAL_WAIT,
            )
            and self.flow_id is not None
            and self.flow_id.startswith("local.")
        )
        if not is_local_flow and (self.flow_id is None) != (self.flow is None):
            raise SchemaError("flow must exactly accompany flow_id", path=f"{path}.flow")
        if is_local_flow and (self.flow is not None or self.flow_route is not None):
            raise SchemaError("local transport cannot carry a cross-die flow/route", path=f"{path}.flow")
        if self.flow is not None:
            self.flow.validate(
                f"{path}.flow",
                allow_equal_ranks_for_state_transfer=isinstance(
                    self.origin_ref, StateTransferOrigin
                ),
            )
            if self.flow.id != self.flow_id:
                raise SchemaError("flow.id disagrees with flow_id", path=f"{path}.flow.id")
        if not is_local_flow and (self.flow_id is None) != (self.flow_route is None):
            raise SchemaError("flow_route must exactly accompany flow_id", path=f"{path}.flow_route")
        if self.flow_route is not None:
            self.flow_route.validate(f"{path}.flow_route")
            if self.flow_route.flow_id != self.flow_id:
                raise SchemaError("flow_route disagrees with flow_id", path=f"{path}.flow_route.flow_id")
        if self.runtime_binding is not None:
            self.runtime_binding.validate(f"{path}.runtime_binding")
            if self.runtime_binding.task_id != self.source.task_id:
                raise SchemaError("runtime_binding disagrees with source task", path=f"{path}.runtime_binding.task_id")
        seen_uses: set[tuple[object, ...]] = set()
        for index, use in enumerate(self.buffer_uses):
            use.validate(f"{path}.buffer_uses[{index}]")
            key = (
                use.binding_id,
                use.access,
                use.role,
                use.operand_index,
                use.contribution_rank,
            )
            if key in seen_uses:
                raise SchemaError("contains a duplicate buffer use", path=f"{path}.buffer_uses[{index}]")
            seen_uses.add(key)
        if self.task_kind is SemanticTaskKind.COMP:
            assert self.compute is not None
            expected_compute_uses = tuple(
                (BufferUseRole.COMP_INPUT, index, BufferAccess.READ)
                for index in range(len(self.compute.inputs))
            ) + tuple(
                (BufferUseRole.COMP_OUTPUT, index, BufferAccess.WRITE)
                for index in range(len(self.compute.outputs))
            )
            actual_compute_uses = tuple(
                (use.role, use.operand_index, use.access)
                for use in self.buffer_uses
            )
            if actual_compute_uses != expected_compute_uses:
                raise SchemaError(
                    "COMP buffer uses must exactly cover ComputeContract roles/arity",
                    path=f"{path}.buffer_uses",
                )
        for index, use in enumerate(self.state_uses):
            use.validate(f"{path}.state_uses[{index}]")
        canonical_state_uses = tuple(
            sorted(
                self.state_uses,
                key=lambda use: (use.hbm_binding_ref, use.access.value),
            )
        )
        if (
            len(
                {
                    (use.hbm_binding_ref, use.access)
                    for use in self.state_uses
                }
            )
            != len(self.state_uses)
        ):
            raise SchemaError(
                "contains a duplicate state use", path=f"{path}.state_uses"
            )
        if self.state_uses != canonical_state_uses:
            raise SchemaError(
                "must use canonical HBM-binding/access order",
                path=f"{path}.state_uses",
            )
        expected_state_access = {
            SemanticTaskKind.DMA_IN: StateUseAccess.READ,
            SemanticTaskKind.DMA_OUT: StateUseAccess.WRITE,
        }.get(self.task_kind)
        is_state_transfer = isinstance(self.origin_ref, StateTransferOrigin)
        if expected_state_access is None:
            if self.state_uses:
                raise SchemaError(
                    "only state DMA may carry state uses",
                    path=f"{path}.state_uses",
                )
            if self.lowering is RegionLowering.STRICT_STATE_IO:
                raise SchemaError(
                    "STRICT_STATE_IO requires a state DMA action",
                    path=f"{path}.lowering",
                )
            if (
                is_state_transfer
                != (self.lowering is RegionLowering.STRICT_STATE_TRANSFER)
            ):
                raise SchemaError(
                    "state-transfer origin and STRICT_STATE_TRANSFER lowering must exactly accompany each other",
                    path=f"{path}.lowering",
                )
        else:
            if self.lowering is not RegionLowering.STRICT_STATE_IO:
                raise SchemaError(
                    "state DMA requires STRICT_STATE_IO lowering",
                    path=f"{path}.lowering",
                )
            if (
                len(self.state_uses) != 1
                or self.state_uses[0].access is not expected_state_access
            ):
                raise SchemaError(
                    "state DMA requires exactly one direction-matching HBM use",
                    path=f"{path}.state_uses",
                )
            expected_local_use = {
                SemanticTaskKind.DMA_IN: (
                    BufferUseRole.DMA_DESTINATION,
                    BufferAccess.WRITE,
                ),
                SemanticTaskKind.DMA_OUT: (
                    BufferUseRole.DMA_SOURCE,
                    BufferAccess.READ,
                ),
            }[self.task_kind]
            if (
                len(self.buffer_uses) != 1
                or (self.buffer_uses[0].role, self.buffer_uses[0].access)
                != expected_local_use
            ):
                raise SchemaError(
                    "state DMA requires exactly one direction-matching local SRAM use",
                    path=f"{path}.buffer_uses",
                )


def _ordered_unique_without_self(values: tuple[str, ...], self_id: str) -> tuple[str, ...]:
    result: list[str] = []
    seen = {self_id}
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GlobalActionDAG:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    source_state_manifest_id: str | None
    source_projection_id: str
    source_schedule_set_id: str
    scheduled_dags: tuple[ScheduledDagRef, ...]
    actions: tuple[GlobalAction, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "GlobalActionDAG":
        return cls(
            schema_version=GLOBAL_ACTION_DAG_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("global_action_dag", semantic_key, schema_version=GLOBAL_ACTION_DAG_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "source_state_manifest_id", "source_projection_id", "source_schedule_set_id", "scheduled_dags", "actions",
        )}

    def validate(self, path: str = "global_action_dag") -> None:
        if self.schema_version != GLOBAL_ACTION_DAG_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        for field_name in ("source_ir1_id", "source_projection_id", "source_schedule_set_id"):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if self.source_state_manifest_id is not None:
            validate_nonempty(
                self.source_state_manifest_id,
                f"{path}.source_state_manifest_id",
            )
        if (
            self.source_state_manifest_id is None
            and any(action.state_uses for action in self.actions)
        ):
            raise SchemaError(
                "state actions require manifest provenance",
                path=f"{path}.source_state_manifest_id",
            )
        if not self.scheduled_dags:
            raise SchemaError("must contain scheduled DAG references", path=f"{path}.scheduled_dags")
        pair_keys: set[tuple[str, str]] = set()
        die_ids: set[int] = set()
        for index, ref in enumerate(self.scheduled_dags):
            ref.validate(f"{path}.scheduled_dags[{index}]")
            key = (ref.dag_id, ref.schedule_id)
            if key in pair_keys or ref.die_id in die_ids:
                raise SchemaError("scheduled DAG pairs and dies must be unique", path=f"{path}.scheduled_dags[{index}]")
            pair_keys.add(key)
            die_ids.add(ref.die_id)
        validate_dependency_dag(self.actions, f"{path}.actions")
        sources: set[tuple[str, str, str]] = set()
        for index, action in enumerate(self.actions):
            action.validate(f"{path}.actions[{index}]")
            pair = (action.source.dag_id, action.source.schedule_id)
            if pair not in pair_keys:
                raise SchemaError("action claims an undeclared DAG/schedule pair", path=f"{path}.actions[{index}].source")
            key = (*pair, action.source.task_id)
            if key in sources:
                raise SchemaError("source task is claimed by multiple actions", path=f"{path}.actions[{index}].source")
            sources.add(key)
        expected_id = stable_artifact_id("global_action_dag", self._semantic_key(), schema_version=GLOBAL_ACTION_DAG_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        ir1: IR1,
        projection: IR2ProjectionResult,
        schedule_set: IntraDieScheduleSet,
        path: str = "global_action_dag",
    ) -> None:
        self.validate(path)
        schedule_set.validate_against(projection, ir1, "intra_die_schedule_set")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("references a different IR-1", path=f"{path}.source_ir1_id")
        if self.source_projection_id != projection.id:
            raise SchemaError("references a different projection", path=f"{path}.source_projection_id")
        if self.source_schedule_set_id != schedule_set.id:
            raise SchemaError("references a different schedule set", path=f"{path}.source_schedule_set_id")
        manifest = ir1.persistent_state_manifest
        expected_manifest_id = manifest.id if manifest is not None else None
        if (
            self.source_state_manifest_id != expected_manifest_id
            or projection.source_state_manifest_id != expected_manifest_id
        ):
            raise SchemaError(
                "persistent-state manifest provenance is not exact",
                path=f"{path}.source_state_manifest_id",
            )
        hbm_binding_index = (
            {binding.id: binding for binding in manifest.bindings}
            if manifest is not None
            else {}
        )
        state_declaration_index = (
            {declaration.id: declaration for declaration in manifest.declarations}
            if manifest is not None
            else {}
        )

        schedules = {schedule.dag_id: schedule for schedule in schedule_set.schedules}
        expected_refs = tuple(
            ScheduledDagRef(schedule.dag_id, schedule.id, schedule.die_id)
            for schedule in schedule_set.schedules
        )
        if self.scheduled_dags != expected_refs:
            raise SchemaError("scheduled DAG references must exactly preserve ScheduleSet pairing/order", path=f"{path}.scheduled_dags")

        action_by_source = {
            (action.source.dag_id, action.source.schedule_id, action.source.task_id): (action, index)
            for index, action in enumerate(self.actions)
        }
        expected_sources = tuple(
            (dag.id, schedules[dag.id].id, task.id)
            for dag in projection.dags
            for task in dag.tasks
        )
        actual_sources = tuple(
            (action.source.dag_id, action.source.schedule_id, action.source.task_id)
            for action in self.actions
        )
        if actual_sources != expected_sources:
            raise SchemaError(
                "actions must exactly preserve projection DAG/task order",
                path=f"{path}.actions",
            )

        source_action_id = {key: item[0].id for key, item in action_by_source.items()}
        wave_task_deps = state_transfer_wave_task_dependencies(
            projection, schedule_set
        )
        for dag in projection.dags:
            schedule = schedules[dag.id]
            region_index = {region.id: region for region in dag.regions}
            flow_index = {flow.id: flow for flow in dag.flows}
            placements = {placement.task_id: placement.core_id for placement in schedule.placements}
            core_orders = {order.core_id: order.task_ids for order in schedule.core_orders}
            die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
            cores = {core.runtime_core_id: core for core in die.cores}
            routes = {route.flow_id: route for route in schedule.flow_routes}
            runtimes = {binding.task_id: binding for binding in schedule.runtime_bindings}
            for task in dag.tasks:
                key = (dag.id, schedule.id, task.id)
                action, action_index = action_by_source[key]
                action_path = f"{path}.actions[{action_index}]"
                semantic = (
                    action.task_kind, action.origin_ref, action.region_id, action.op_kind,
                    action.member_id, action.flow_id, action.chunk_id, action.collective_step,
                    action.source_rank, action.destination_rank, action.tensor_slice, action.bytes,
                    action.dtype, action.shape, action.read_values, action.write_values,
                    action.compute, action.reduction, action.sync, action.dma,
                )
                expected_semantic = (
                    task.kind, task.origin_ref, task.region_id, task.op_kind, task.member_id,
                    task.flow_id, task.chunk_id, task.collective_step, task.source_rank,
                    task.destination_rank, task.tensor_slice, task.bytes, task.dtype, task.shape,
                    task.read_values, task.write_values, task.compute, task.reduction, task.sync, task.dma,
                )
                if semantic != expected_semantic:
                    raise SchemaError("action does not exactly preserve source task semantics", path=action_path)
                if task.region_id is None or task.region_id not in region_index:
                    raise SchemaError("source task must belong to exactly one lowering region", path=f"{action_path}.region_id")
                if action.lowering is not region_index[task.region_id].lowering:
                    raise SchemaError("lowering disagrees with source region", path=f"{action_path}.lowering")
                expected_flow = flow_index.get(task.flow_id) if task.flow_id is not None else None
                expected_route = routes.get(task.flow_id) if task.flow_id is not None else None
                expected_runtime = runtimes.get(task.id)
                if action.flow != expected_flow:
                    raise SchemaError("flow/channel/payload is not exact", path=f"{action_path}.flow")
                if action.flow_route != expected_route:
                    raise SchemaError("flow route is not exact", path=f"{action_path}.flow_route")
                if action.runtime_binding != expected_runtime:
                    raise SchemaError("runtime binding is not exact", path=f"{action_path}.runtime_binding")
                expected_uses = tuple(
                    ActionBufferUse(
                        use.binding_id,
                        use.access,
                        use.role,
                        use.operand_index,
                        use.contribution_rank,
                        use.tensor_slice,
                    )
                    for use in schedule.task_buffer_uses if use.task_id == task.id
                )
                if action.buffer_uses != expected_uses:
                    raise SchemaError("buffer uses are not exact", path=f"{action_path}.buffer_uses")
                expected_state_uses = tuple(
                    ActionStateUse(
                        use.hbm_binding_ref,
                        use.access,
                    )
                    for use in schedule.task_state_uses
                    if use.task_id == task.id
                )
                if action.state_uses != expected_state_uses:
                    raise SchemaError(
                        "state uses are not an exact schedule quotient",
                        path=f"{action_path}.state_uses",
                    )
                if any(
                    use.binding_id in hbm_binding_index
                    for use in action.buffer_uses
                ):
                    raise SchemaError(
                        "HBM bindings cannot be represented as SRAM buffer uses",
                        path=f"{action_path}.buffer_uses",
                    )
                for state_use_index, state_use in enumerate(action.state_uses):
                    binding = hbm_binding_index.get(state_use.hbm_binding_ref)
                    if binding is None:
                        raise SchemaError(
                            "state use references an unknown HBM binding",
                            path=(
                                f"{action_path}.state_uses"
                                f"[{state_use_index}].hbm_binding_ref"
                            ),
                        )
                    declaration = state_declaration_index[binding.state_ref]
                    if (
                        task.dma is None
                        or binding.state_ref != task.dma.state_ref
                        or binding.die_id != dag.die_id
                        or declaration.access is PersistentStateAccess.RESERVED
                    ):
                        raise SchemaError(
                            "state use HBM binding/state/die are not exact",
                            path=f"{action_path}.state_uses[{state_use_index}]",
                        )
                    if (
                        task.dma.state_offset_bytes > binding.size_bytes
                        or task.bytes
                        > binding.size_bytes - task.dma.state_offset_bytes
                    ):
                        raise SchemaError(
                            "state use DMA byte range exceeds its HBM binding",
                            path=f"{action_path}.state_uses[{state_use_index}]",
                        )

                predecessor: tuple[str, ...] = ()
                if task.kind is SemanticTaskKind.TRANSIT:
                    expected_core = None
                    expected_position = None
                else:
                    runtime_core = placements[task.id]
                    core = cores[runtime_core]
                    expected_core = LogicalCoreRef(dag.die_id, core.local_core_id)
                    order = core_orders[runtime_core]
                    expected_position = order.index(task.id)
                    if expected_position:
                        predecessor_task = order[expected_position - 1]
                        predecessor = (source_action_id[(dag.id, schedule.id, predecessor_task)],)
                if action.logical_core != expected_core:
                    raise SchemaError("logical core does not match runtime placement", path=f"{action_path}.logical_core")
                if action.core_order_index != expected_position:
                    raise SchemaError("core order index is not exact", path=f"{action_path}.core_order_index")
                semantic_deps = tuple(source_action_id[(dag.id, schedule.id, dep)] for dep in task.deps)
                wave_deps = tuple(
                    source_action_id[
                        (
                            source_dag,
                            schedules[source_dag].id,
                            source_task,
                        )
                    ]
                    for source_dag, source_task in wave_task_deps.get(
                        (dag.id, task.id), ()
                    )
                )
                expected_deps = _ordered_unique_without_self(
                    semantic_deps + predecessor + wave_deps, action.id
                )
                if action.deps != expected_deps:
                    raise SchemaError("deps must exactly equal semantic deps, same-core predecessor, and state-transfer wave edge", path=f"{action_path}.deps")

        dag_die = {ref.dag_id: ref.die_id for ref in self.scheduled_dags}
        local_action_index = {
            (dag_die[action.source.dag_id], action.source.task_id): action
            for action in self.actions
        }
        transfer_actions: dict[
            tuple[str, int, SemanticTaskKind], list[GlobalAction]
        ] = {}
        state_actions: dict[
            tuple[str, int, SemanticTaskKind], list[GlobalAction]
        ] = {}
        for action in self.actions:
            die_id = dag_die[action.source.dag_id]
            if isinstance(action.origin_ref, StateTransferOrigin):
                transfer_actions.setdefault(
                    (
                        action.origin_ref.state_transfer_ref,
                        die_id,
                        action.task_kind,
                    ),
                    [],
                ).append(action)
            elif isinstance(action.origin_ref, StateIoOrigin):
                state_actions.setdefault(
                    (
                        action.origin_ref.state_access_ref,
                        die_id,
                        action.task_kind,
                    ),
                    [],
                ).append(action)

        access_index = {access.id: access for access in ir1.state_accesses}
        declaration_index = (
            {declaration.id: declaration for declaration in manifest.declarations}
            if manifest is not None
            else {}
        )
        pair_routes = {
            route.id: route
            for group in ir1.groups
            for route in group.embedding.routes
        }
        cross_routes = {route.id: route for route in ir1.cross_routes}

        def one_action(
            matches: list[GlobalAction] | None,
            *,
            message: str,
        ) -> GlobalAction:
            if matches is None or len(matches) != 1:
                raise SchemaError(message, path=f"{path}.actions")
            return matches[0]

        def one_buffer(
            action: GlobalAction, role: BufferUseRole
        ) -> ActionBufferUse:
            matches = tuple(
                use for use in action.buffer_uses if use.role is role
            )
            if len(matches) != 1:
                raise SchemaError(
                    "state-transfer action requires one exact staging buffer use",
                    path=f"{path}.actions",
                )
            return matches[0]

        def one_transfer_action(
            matches: list[GlobalAction] | None,
            *,
            segment_index: int | None,
            message: str,
        ) -> GlobalAction:
            candidates = tuple(
                action
                for action in matches or ()
                if isinstance(action.origin_ref, StateTransferOrigin)
                and action.origin_ref.segment_index == segment_index
            )
            if len(candidates) != 1:
                raise SchemaError(message, path=f"{path}.actions")
            return candidates[0]

        for contract in projection.state_transfers:
            source_access = access_index[contract.source_state_access_ref]
            destination_access = access_index[
                contract.destination_state_access_ref
            ]
            source_declaration = declaration_index[source_access.state_ref]
            destination_declaration = declaration_index[
                destination_access.state_ref
            ]
            if isinstance(contract, SegmentedKvStateTransferContract):
                route = cross_routes[contract.cross_group_route_ref]
                source_dma_kind = SemanticTaskKind.DMA_OUT
                source_dma_role = BufferUseRole.DMA_SOURCE
                source_state_use = StateUseAccess.WRITE
                transfer_units = tuple(
                    (
                        segment_index,
                        segment.source_local_offset,
                        segment.source_local_shape,
                        segment.destination_local_offset,
                        segment.destination_local_shape,
                        segment.bytes,
                    )
                    for segment_index, segment in enumerate(contract.segments)
                )
            elif isinstance(contract, SlicedKvStateTransferContract):
                route = cross_routes[contract.cross_group_route_ref]
                source_dma_kind = SemanticTaskKind.DMA_OUT
                source_dma_role = BufferUseRole.DMA_SOURCE
                source_state_use = StateUseAccess.WRITE
                transfer_units = ((
                    None,
                    contract.source_local_offset,
                    contract.source_local_shape,
                    contract.destination_local_offset,
                    contract.destination_local_shape,
                    contract.bytes,
                ),)
            elif isinstance(contract, StateTransferContract):
                route = pair_routes[contract.pair_route_ref]
                source_dma_kind = SemanticTaskKind.DMA_IN
                source_dma_role = BufferUseRole.DMA_DESTINATION
                source_state_use = StateUseAccess.READ
                transfer_units = ((
                    None,
                    (0,) * len(source_declaration.shape),
                    source_declaration.shape,
                    (0,) * len(destination_declaration.shape),
                    destination_declaration.shape,
                    source_declaration.tensor_bytes,
                ),)
            else:
                raise SchemaError(
                    "projection contains an unknown state-transfer contract",
                    path=f"{path}.actions",
                )
            source_die = route.die_path[0]
            destination_die = route.die_path[-1]
            source_dma = one_action(
                state_actions.get(
                    (
                        source_access.id,
                        source_die,
                        source_dma_kind,
                    )
                ),
                message="state-transfer source requires one direction-exact state DMA action",
            )
            destination_dma = one_action(
                state_actions.get(
                    (
                        destination_access.id,
                        destination_die,
                        SemanticTaskKind.DMA_OUT,
                    )
                ),
                message="state-transfer destination requires one DMA_OUT action",
            )
            assert source_dma.dma is not None
            assert destination_dma.dma is not None
            source_targets = tuple(
                local_action_index[(source_die, task_id)].id
                for task_id in source_dma.dma.access_task_refs
            )
            destination_targets = tuple(
                local_action_index[(destination_die, task_id)].id
                for task_id in destination_dma.dma.access_task_refs
            )
            if (
                len(source_dma.state_uses) != 1
                or source_dma.state_uses[0].access is not source_state_use
                or len(destination_dma.state_uses) != 1
                or destination_dma.state_uses[0].access
                is not StateUseAccess.WRITE
            ):
                raise SchemaError(
                    "state-transfer action HBM direction is not exact",
                    path=f"{path}.actions",
                )
            for (
                segment_index,
                source_offset,
                source_shape,
                destination_offset,
                destination_shape,
                payload_bytes,
            ) in transfer_units:
                send = one_transfer_action(
                    transfer_actions.get(
                        (contract.id, source_die, SemanticTaskKind.SEND)
                    ),
                    segment_index=segment_index,
                    message="state-transfer unit requires one source SEND action",
                )
                recv = one_transfer_action(
                    transfer_actions.get(
                        (contract.id, destination_die, SemanticTaskKind.RECV)
                    ),
                    segment_index=segment_index,
                    message="state-transfer unit requires one destination RECV action",
                )
                wait = one_transfer_action(
                    transfer_actions.get(
                        (contract.id, destination_die, SemanticTaskKind.WAIT)
                    ),
                    segment_index=segment_index,
                    message="state-transfer unit requires one destination WAIT action",
                )
                source_chain = (
                    all(target in send.deps for target in source_targets)
                    and (
                        all(
                            target in source_dma.deps
                            for target in source_targets
                        )
                        if source_dma_kind is SemanticTaskKind.DMA_OUT
                        else all(
                            source_dma.id
                            in local_action_index[(source_die, task_id)].deps
                            for task_id in source_dma.dma.access_task_refs
                        )
                    )
                )
                destination_chain = (
                    recv.id in wait.deps
                    and all(
                        wait.id
                        in local_action_index[(destination_die, task_id)].deps
                        for task_id in destination_dma.dma.access_task_refs
                    )
                    and all(
                        target in destination_dma.deps
                        for target in destination_targets
                    )
                )
                if not source_targets or not source_chain:
                    raise SchemaError(
                        "state-transfer source action dependency chain is not exact",
                        path=f"{path}.actions",
                    )
                if not destination_targets or not destination_chain:
                    raise SchemaError(
                        "state-transfer destination action dependency chain is not exact",
                        path=f"{path}.actions",
                    )
                if (
                    send.flow is None
                    or recv.flow is None
                    or send.flow_route is None
                    or recv.flow_route is None
                    or send.flow.pair_route_ref != route.id
                    or recv.flow.pair_route_ref != route.id
                    or send.flow_route.pair_route_ref != route.id
                    or recv.flow_route.pair_route_ref != route.id
                    or send.bytes != payload_bytes
                    or recv.bytes != payload_bytes
                    or send.tensor_slice is None
                    or recv.tensor_slice is None
                    or send.tensor_slice.offset != source_offset
                    or send.tensor_slice.shape != source_shape
                    or recv.tensor_slice.offset != destination_offset
                    or recv.tensor_slice.shape != destination_shape
                ):
                    raise SchemaError(
                        "state-transfer action route/payload does not match its contract",
                        path=f"{path}.actions",
                    )
                if (
                    one_buffer(send, BufferUseRole.SEND_SOURCE).binding_id
                    != one_buffer(source_dma, source_dma_role).binding_id
                    or one_buffer(
                        recv, BufferUseRole.RECV_DESTINATION
                    ).binding_id
                    != one_buffer(
                        destination_dma, BufferUseRole.DMA_SOURCE
                    ).binding_id
                ):
                    raise SchemaError(
                        "state-transfer action staging lineage is not exact",
                        path=f"{path}.actions",
                    )
                for transit_die in route.die_path[1:-1]:
                    transit = one_transfer_action(
                        transfer_actions.get(
                            (
                                contract.id,
                                transit_die,
                                SemanticTaskKind.TRANSIT,
                            )
                        ),
                        segment_index=segment_index,
                        message=(
                            "state-transfer unit requires one TRANSIT action "
                            "on every intermediate die"
                        ),
                    )
                    if (
                        transit.flow is None
                        or transit.flow_route is None
                        or transit.flow.pair_route_ref != route.id
                        or transit.flow_route.pair_route_ref != route.id
                        or transit.bytes != payload_bytes
                        or transit.logical_core is not None
                        or transit.core_order_index is not None
                        or transit.runtime_binding is not None
                        or transit.buffer_uses
                        or transit.state_uses
                    ):
                        raise SchemaError(
                            "TRANSIT action route/resource quotient is not exact",
                            path=f"{path}.actions",
                        )
