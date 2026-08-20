"""Deterministic, fail-closed N5 intra-die scheduling policy."""

from __future__ import annotations

import heapq
import math

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.ir1 import IR1, MemoryInitiator
from ..schema.ir0 import OpKind
from ..schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    FlowRouteBinding,
    FlowRouteRole,
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieSchedule,
    IntraDieScheduleSet,
    LogicalRuntimeBinding,
    OrdinaryNodeOrigin,
    PortLeg,
    SemanticTask,
    SemanticTaskKind,
    StateTransferOrigin,
    StateUseAccess,
    TaskBufferUse,
    TaskPlacement,
    TaskStateUse,
    TensorSlice,
    dense_row_major_view_byte_addend,
)
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateLifetime,
    StateKind,
)


_USE_ROLE_ORDER = {
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
NAIVE_INTRADIE_POLICY_SCHEMA_VERSION = (
    "wafer_frontend.naive_intra_die_policy/v8"
)
_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _canonical_kahn(dag: IntraDieDAG) -> tuple[SemanticTask, ...]:
    """Topological order with source tuple position as the primary tie-break."""

    task_index = {task.id: task for task in dag.tasks}
    source_index = {task.id: index for index, task in enumerate(dag.tasks)}
    indegree = {task.id: len(task.deps) for task in dag.tasks}
    successors: dict[str, list[str]] = {task.id: [] for task in dag.tasks}
    for task in dag.tasks:
        for dependency in task.deps:
            successors[dependency].append(task.id)
    ready = [
        (source_index[task.id], task.id)
        for task in dag.tasks
        if indegree[task.id] == 0
    ]
    heapq.heapify(ready)
    result: list[SemanticTask] = []
    while ready:
        _index, task_id = heapq.heappop(ready)
        result.append(task_index[task_id])
        for successor in successors[task_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(
                    ready,
                    (source_index[successor], successor),
                )
    if len(result) != len(dag.tasks):
        _fail("task dependency graph contains a cycle", "projection.dags")
    return tuple(result)


def _executable_components(
    topological: tuple[SemanticTask, ...],
) -> tuple[tuple[str, ...], ...]:
    executable = {
        task.id
        for task in topological
        if task.kind is not SemanticTaskKind.TRANSIT
    }
    neighbors: dict[str, set[str]] = {task_id: set() for task_id in executable}
    for task in topological:
        if task.id not in executable:
            continue
        for dependency in task.deps:
            if dependency in executable:
                neighbors[task.id].add(dependency)
                neighbors[dependency].add(task.id)
    position = {task.id: index for index, task in enumerate(topological)}
    unseen = set(executable)
    result: list[tuple[str, ...]] = []
    while unseen:
        root = min(unseen, key=lambda task_id: (position[task_id], task_id))
        stack = [root]
        members: set[str] = set()
        unseen.remove(root)
        while stack:
            current = stack.pop()
            members.add(current)
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(
            tuple(sorted(members, key=lambda item: (position[item], item)))
        )
    result.sort(key=lambda members: (position[members[0]], members[0]))
    return tuple(result)


def _banks(
    absolute_start: int,
    size_bytes: int,
    *,
    bank_count: int,
    interleave_bytes: int,
) -> tuple[int, ...]:
    first_stripe = absolute_start // interleave_bytes
    last_stripe = (absolute_start + size_bytes - 1) // interleave_bytes
    stripe_count = last_stripe - first_stripe + 1
    if stripe_count >= bank_count:
        return tuple(range(bank_count))
    return tuple(
        sorted(
            (first_stripe + index) % bank_count
            for index in range(stripe_count)
        )
    )


def _xy_path(
    source: tuple[int, int],
    destination: tuple[int, int],
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


def _component_cores(
    component: tuple[str, ...],
    task_by_id: dict[str, SemanticTask],
    cores: tuple[object, ...],
    profile_index: dict[str, object],
):
    initiator_by_kind = {
        SemanticTaskKind.COMP: MemoryInitiator.COMPUTE,
        SemanticTaskKind.SEND: MemoryInitiator.DTE,
        SemanticTaskKind.RECV: MemoryInitiator.NOC_RX,
        SemanticTaskKind.REDUCE: MemoryInitiator.COMPUTE,
        SemanticTaskKind.LOCAL_COPY: MemoryInitiator.LSU,
        SemanticTaskKind.DMA_IN: MemoryInitiator.LSU,
        SemanticTaskKind.DMA_OUT: MemoryInitiator.LSU,
    }
    initiators = {
        initiator_by_kind[task_by_id[task_id].kind]
        for task_id in component
        if task_by_id[task_id].kind in initiator_by_kind
    }
    return tuple(
        core
        for core in cores
        if any(
            initiators.issubset(region.access)
            for region in profile_index[core.sram_profile_ref].regions
        )
    )


def _ordinary_rank_local_view(
    task: SemanticTask,
    value: object,
    ir1: IR1,
) -> TensorSlice:
    """Derive one ordinary operand's exact rank-local logical domain."""

    if not isinstance(task.origin_ref, OrdinaryNodeOrigin):
        _fail(
            "non-tiled COMP requires an ordinary rank origin",
            f"projection.dags.tasks.{task.id}.origin_ref",
        )
    node = next(
        (candidate for candidate in ir1.nodes if candidate.id == task.origin_ref.op_id),
        None,
    )
    if node is None or task.member_id != node.id:
        _fail(
            "ordinary task does not identify its exact IR1 node",
            f"projection.dags.tasks.{task.id}.member_id",
        )
    group = next(
        (
            candidate
            for candidate in ir1.groups
            if candidate.id == node.execution_group_ref
        ),
        None,
    )
    if group is None:
        _fail(
            "ordinary node references an unknown execution group",
            f"ir1.nodes.{node.id}.execution_group_ref",
        )
    placement = next(
        (
            candidate
            for candidate in group.placements
            if candidate.rank == task.origin_ref.rank
        ),
        None,
    )
    if placement is None:
        _fail(
            "ordinary rank has no exact group placement",
            f"projection.dags.tasks.{task.id}.origin_ref.rank",
        )
    if len(group.logical_shape) != 1 or len(placement.logical_coord) != 1:
        _fail(
            "naive ordinary sharding requires a one-axis physical group",
            f"ir1.groups.{group.id}.logical_shape",
        )
    mapped_dimensions = tuple(
        axis
        for axis, mesh_axis in enumerate(value.sharding.dim_map)
        if mesh_axis is group.axis
    )
    if len(mapped_dimensions) > 1:
        _fail(
            "one physical group axis cannot shard multiple tensor dimensions",
            f"projection.dags.values.{value.id}.sharding.dim_map",
        )
    offset = [0] * len(value.shape)
    shape = list(value.shape)
    if mapped_dimensions:
        if value.sharding.mesh_ref != group.mesh_ref:
            _fail(
                "rank-local sharding mesh disagrees with the execution group",
                f"projection.dags.values.{value.id}.sharding.mesh_ref",
            )
        tensor_axis = mapped_dimensions[0]
        degree = group.logical_shape[0]
        if value.shape[tensor_axis] % degree:
            _fail(
                "rank-local tensor extent must divide its physical group degree",
                f"projection.dags.values.{value.id}.shape[{tensor_axis}]",
            )
        extent = value.shape[tensor_axis] // degree
        coordinate = placement.logical_coord[0]
        offset[tensor_axis] = coordinate * extent
        shape[tensor_axis] = extent
    return TensorSlice(value.id, tuple(offset), tuple(shape))


def _minimum_root(
    value_id: str,
    views: tuple[TensorSlice, ...],
    dtype: DType,
) -> TensorSlice:
    if not views or any(
        view.value_id != value_id or len(view.shape) != len(views[0].shape)
        for view in views
    ):
        _fail("buffer views do not share one value/rank", "schedule.buffer_views")
    rank = len(views[0].shape)
    minimum = tuple(min(view.offset[axis] for view in views) for axis in range(rank))
    maximum = tuple(
        max(view.offset[axis] + view.shape[axis] for view in views)
        for axis in range(rank)
    )
    root = TensorSlice(
        value_id,
        minimum,
        tuple(maximum[axis] - minimum[axis] for axis in range(rank)),
    )
    for index, view in enumerate(views):
        dense_row_major_view_byte_addend(
            root,
            view,
            dtype,
            path=f"schedule.buffer_views[{index}]",
        )
    return root


def _ordinary_schedule(dag: IntraDieDAG, ir1: IR1) -> IntraDieSchedule:
    topological = _canonical_kahn(dag)
    supported = {
        SemanticTaskKind.COMP,
        SemanticTaskKind.SEND,
        SemanticTaskKind.RECV,
        SemanticTaskKind.REDUCE,
        SemanticTaskKind.LOCAL_COPY,
        SemanticTaskKind.WAIT,
        SemanticTaskKind.BARRIER,
        SemanticTaskKind.TRANSIT,
        SemanticTaskKind.DMA_IN,
        SemanticTaskKind.DMA_OUT,
    }
    unsupported = tuple(
        task.kind
        for task in topological
        if task.kind not in supported
    )
    if unsupported:
        _fail(
            "naive scheduler does not support this task kind yet",
            "projection.dags.tasks",
        )
    die = next(item for item in ir1.fabric.dies if item.id == dag.die_id)
    cores = tuple(sorted(die.cores, key=lambda core: core.runtime_core_id))
    profile_index = {
        profile.id: profile for profile in ir1.fabric.sram_profiles
    }
    if not cores and any(
        task.kind is not SemanticTaskKind.TRANSIT for task in topological
    ):
        _fail("die has no executable core", "ir1.fabric.dies.cores")
    components = _executable_components(topological)
    placement_by_task: dict[str, int] = {}
    next_core_by_initiators: dict[frozenset[MemoryInitiator], int] = {}
    task_by_id = {task.id: task for task in topological}
    for component in components:
        compatible_cores = _component_cores(
            component,
            task_by_id,
            cores,
            profile_index,
        )
        if not compatible_cores:
            _fail(
                "component has no core with one SRAM region for all required initiators",
                "ir1.fabric.sram_profiles.regions",
            )
        signature = frozenset(
            initiator
            for task_id in component
            for initiator in (
                {
                    SemanticTaskKind.COMP: MemoryInitiator.COMPUTE,
                    SemanticTaskKind.SEND: MemoryInitiator.DTE,
                    SemanticTaskKind.RECV: MemoryInitiator.NOC_RX,
                    SemanticTaskKind.REDUCE: MemoryInitiator.COMPUTE,
                    SemanticTaskKind.LOCAL_COPY: MemoryInitiator.LSU,
                    SemanticTaskKind.DMA_IN: MemoryInitiator.LSU,
                    SemanticTaskKind.DMA_OUT: MemoryInitiator.LSU,
                }.get(task_by_id[task_id].kind),
            )
            if initiator is not None
        )
        next_index = next_core_by_initiators.get(signature, 0)
        core_id = compatible_cores[
            next_index % len(compatible_cores)
        ].runtime_core_id
        next_core_by_initiators[signature] = next_index + 1
        for task_id in component:
            placement_by_task[task_id] = core_id
    placements = tuple(
        TaskPlacement(task.id, placement_by_task[task.id])
        for task in topological
        if task.kind is not SemanticTaskKind.TRANSIT
    )
    order_by_core = {
        core.runtime_core_id: tuple(
            task.id
            for task in topological
            if placement_by_task.get(task.id) == core.runtime_core_id
        )
        for core in cores
    }
    core_orders = tuple(
        CoreOrder(core.runtime_core_id, order_by_core[core.runtime_core_id])
        for core in cores
        if order_by_core[core.runtime_core_id]
    )
    position_by_task = {
        task_id: position
        for task_ids in order_by_core.values()
        for position, task_id in enumerate(task_ids)
    }
    value_index = {
        value.id: value
        for value in (*dag.values, *dag.state_staging_values)
    }
    state_value_ids = {value.id for value in dag.state_staging_values}
    staging_value_index = {
        value.id: value for value in dag.state_staging_values
    }
    alias_source_by_value: dict[str, str] = {}
    for value in dag.values:
        if value.alias_set is None:
            continue
        if len(value.producer_tasks) != 1:
            _fail(
                "aliased value requires one exact producer",
                f"projection.dags.values.{value.id}.producer_tasks",
            )
        producer = task_by_id[value.producer_tasks[0]]
        if (
            producer.kind is not SemanticTaskKind.COMP
            or producer.op_kind is not OpKind.OPTIMIZER_UPDATE
            or producer.compute is None
            or producer.compute.effects.alias_set != value.alias_set
            or tuple(operand.role for operand in producer.compute.inputs)
            != ("weight", "weight_gradient")
            or tuple(operand.role for operand in producer.compute.outputs)
            != ("updated_weight",)
            or producer.compute.outputs[0].value_id != value.id
        ):
            _fail(
                "only exact OPTIMIZER_UPDATE output aliasing is supported",
                f"projection.dags.values.{value.id}.alias_set",
            )
        source_id = producer.compute.inputs[0].value_id
        staging = staging_value_index.get(source_id)
        manifest = ir1.persistent_state_manifest
        if staging is None or manifest is None:
            _fail(
                "optimizer alias root must be persistent-state staging",
                f"projection.dags.values.{value.id}.alias_set",
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
        if (
            declaration is None
            or access is None
            or declaration.identity.kind is not StateKind.TRAINABLE_PARAMETER
            or declaration.lifetime is not PersistentStateLifetime.PERSISTENT
            or declaration.access is not PersistentStateAccess.READ_WRITE
            or declaration.identity.tensor_ref is None
            or value.alias_set
            != f"trainable:{declaration.identity.tensor_ref}"
            or access.state_ref != declaration.id
            or access.node_ref != producer.member_id
            or value.shape != staging.shape
            or value.dtype is not staging.dtype
            or value.logical_layout != staging.logical_layout
        ):
            _fail(
                "optimizer alias root must be the exact READ_WRITE trainable state",
                f"projection.dags.values.{value.id}.alias_set",
            )
        alias_source_by_value[value.id] = source_id
    state_domain_by_value = {
        value.id: TensorSlice(
            value.id,
            (0,) * len(value.shape),
            value.shape,
        )
        for value in dag.state_staging_values
    }
    state_view_by_value: dict[str, TensorSlice] = {}
    for task in topological:
        if task.kind not in (
            SemanticTaskKind.DMA_IN,
            SemanticTaskKind.DMA_OUT,
        ):
            continue
        assert task.dma is not None
        assert task.tensor_slice is not None
        state_domain = state_domain_by_value.get(task.dma.local_value_ref)
        if state_domain is None:
            _fail(
                "state DMA references an unknown staging value",
                f"projection.dags.tasks.{task.id}.tensor_slice",
            )
        old_view = state_view_by_value.setdefault(
            task.dma.local_value_ref,
            task.tensor_slice,
        )
        if old_view != task.tensor_slice:
            for view in (old_view, task.tensor_slice):
                dense_row_major_view_byte_addend(
                    state_domain,
                    view,
                    value_index[task.dma.local_value_ref].dtype,
                    path=f"projection.dags.tasks.{task.id}.tensor_slice",
                )
    if set(state_view_by_value) != state_value_ids:
        _fail(
            "state staging values must have exactly one canonical DMA view",
            "projection.dags.state_staging_values",
        )
    required: list[
        tuple[str, BufferUseRole, int, int | None, str, BufferAccess]
    ] = []
    tile_slices: dict[tuple[str, BufferUseRole, int], TensorSlice] = {}
    for task in topological:
        if task.kind is not SemanticTaskKind.COMP:
            continue
        assert task.compute is not None
        required.extend(
            (
                task.id,
                BufferUseRole.COMP_INPUT,
                operand_index,
                None,
                operand.value_id,
                BufferAccess.READ,
            )
            for operand_index, operand in enumerate(task.compute.inputs)
        )
        required.extend(
            (
                task.id,
                BufferUseRole.COMP_OUTPUT,
                operand_index,
                None,
                operand.value_id,
                BufferAccess.WRITE,
            )
            for operand_index, operand in enumerate(task.compute.outputs)
        )
        if task.compute.tile is not None:
            tile_slices.update(
                {
                    (task.id, BufferUseRole.COMP_INPUT, operand_index): TensorSlice(
                        binding.operand_id,
                        binding.logical_offset,
                        binding.logical_shape,
                    )
                    for operand_index, binding in enumerate(
                        task.compute.tile.input_slices
                    )
                }
            )
            tile_slices.update(
                {
                    (task.id, BufferUseRole.COMP_OUTPUT, operand_index): TensorSlice(
                        binding.operand_id,
                        binding.logical_offset,
                        binding.logical_shape,
                    )
                    for operand_index, binding in enumerate(
                        task.compute.tile.output_slices
                    )
                }
            )
    for task in topological:
        if task.kind is SemanticTaskKind.SEND:
            required.extend(
                (
                    task.id,
                    BufferUseRole.SEND_SOURCE,
                    operand_index,
                    None,
                    value_id,
                    BufferAccess.READ,
                )
                for operand_index, value_id in enumerate(task.read_values)
            )
        elif task.kind is SemanticTaskKind.RECV:
            required.extend(
                (
                    task.id,
                    BufferUseRole.RECV_DESTINATION,
                    operand_index,
                    None,
                    value_id,
                    BufferAccess.WRITE,
                )
                for operand_index, value_id in enumerate(task.write_values)
            )
        elif task.kind is SemanticTaskKind.REDUCE:
            assert task.reduction is not None
            required.extend(
                (
                    task.id,
                    BufferUseRole.REDUCE_INPUT,
                    operand_index,
                    rank,
                    task.read_values[operand_index],
                    BufferAccess.READ,
                )
                for operand_index, rank in enumerate(
                    task.reduction.input_ranks
                )
            )
            required.extend(
                (
                    task.id,
                    BufferUseRole.REDUCE_OUTPUT,
                    operand_index,
                    None,
                    value_id,
                    BufferAccess.WRITE,
                )
                for operand_index, value_id in enumerate(task.write_values)
            )
        elif task.kind is SemanticTaskKind.LOCAL_COPY:
            required.extend(
                (
                    task.id,
                    BufferUseRole.LOCAL_COPY_SOURCE,
                    operand_index,
                    None,
                    value_id,
                    BufferAccess.READ,
                )
                for operand_index, value_id in enumerate(task.read_values)
            )
            required.extend(
                (
                    task.id,
                    BufferUseRole.LOCAL_COPY_DESTINATION,
                    operand_index,
                    None,
                    value_id,
                    BufferAccess.WRITE,
                )
                for operand_index, value_id in enumerate(task.write_values)
            )
        elif task.kind is SemanticTaskKind.DMA_IN:
            assert task.dma is not None
            required.append(
                (
                    task.id,
                    BufferUseRole.DMA_DESTINATION,
                    0,
                    None,
                    task.dma.local_value_ref,
                    BufferAccess.WRITE,
                )
            )
        elif task.kind is SemanticTaskKind.DMA_OUT:
            assert task.dma is not None
            required.append(
                (
                    task.id,
                    BufferUseRole.DMA_SOURCE,
                    0,
                    None,
                    task.dma.local_value_ref,
                    BufferAccess.READ,
                )
            )

    uses_by_key: dict[
        tuple[int, str],
        list[tuple[str, BufferAccess, BufferUseRole, TensorSlice]],
    ] = {}
    requirement_keys: list[
        tuple[
            tuple[int, str],
            str,
            BufferUseRole,
            int,
            int | None,
            BufferAccess,
            TensorSlice,
        ]
    ] = []
    for task_id, role, operand_index, contribution_rank, value_id, access in required:
        value = value_index[value_id]
        task = task_by_id[task_id]
        tensor_slice = tile_slices.get((task_id, role, operand_index))
        if value_id in state_value_ids:
            exact_state_view = state_view_by_value[value_id]
            if task.kind in (
                SemanticTaskKind.DMA_IN,
                SemanticTaskKind.DMA_OUT,
            ):
                assert task.tensor_slice is not None
                tensor_slice = task.tensor_slice
            elif (
                task.kind
                in (SemanticTaskKind.SEND, SemanticTaskKind.RECV)
                and isinstance(task.origin_ref, StateTransferOrigin)
            ):
                assert task.tensor_slice is not None
                tensor_slice = task.tensor_slice
            elif tensor_slice is None:
                tensor_slice = exact_state_view
            elif tensor_slice != exact_state_view:
                _fail(
                    "fused parameter tile must equal its exact DMA view",
                    f"projection.dags.tasks.{task_id}.compute.tile",
                )
        elif tensor_slice is None:
            if (
                task.kind
                in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                    SemanticTaskKind.REDUCE,
                    SemanticTaskKind.LOCAL_COPY,
                    SemanticTaskKind.DMA_IN,
                    SemanticTaskKind.DMA_OUT,
                )
                and task.tensor_slice is not None
            ):
                tensor_slice = TensorSlice(
                    value_id,
                    task.tensor_slice.offset,
                    task.tensor_slice.shape,
                )
            elif task.kind is SemanticTaskKind.COMP:
                tensor_slice = _ordinary_rank_local_view(task, value, ir1)
            else:
                tensor_slice = TensorSlice(
                    value_id,
                    (0,) * len(value.shape),
                    value.shape,
                )
        key = (placement_by_task[task_id], value_id)
        uses_by_key.setdefault(key, []).append(
            (task_id, access, role, tensor_slice)
        )
        requirement_keys.append(
            (
                key,
                task_id,
                role,
                operand_index,
                contribution_rank,
                access,
                tensor_slice,
            )
        )

    root_by_key = {
        key: _minimum_root(
            key[1],
            tuple(view for _task, _access, _role, view in accesses),
            value_index[key[1]].dtype,
        )
        for key, accesses in uses_by_key.items()
    }
    alias_root_key_by_key: dict[tuple[int, str], tuple[int, str]] = {}
    for key in uses_by_key:
        source_value_id = alias_source_by_value.get(key[1])
        if source_value_id is None:
            continue
        source_key = (key[0], source_value_id)
        source_root = root_by_key.get(source_key)
        alias_root = root_by_key[key]
        source_value = value_index.get(source_value_id)
        alias_value = value_index[key[1]]
        if (
            source_root is None
            or source_value is None
            or source_root.offset != alias_root.offset
            or source_root.shape != alias_root.shape
            or source_value.dtype is not alias_value.dtype
            or source_value.logical_layout != alias_value.logical_layout
        ):
            _fail(
                "optimizer alias output and state root require one exact view",
                f"schedule.buffer_views.{key[1]}",
            )
        alias_root_key_by_key[key] = source_key

    cursor_by_core_region: dict[tuple[int, str], int] = {}
    binding_by_key: dict[tuple[int, str], BufferBinding] = {}
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
    communication_roles = {
        BufferUseRole.SEND_SOURCE,
        BufferUseRole.RECV_DESTINATION,
        BufferUseRole.REDUCE_INPUT,
        BufferUseRole.REDUCE_OUTPUT,
    }
    forced_region_by_key: dict[tuple[int, str], object] = {}
    allocation_keys: list[tuple[int, str]] = []
    for task in topological:
        if task.kind is not SemanticTaskKind.REDUCE:
            continue
        group_requirements = tuple(
            item
            for item in requirement_keys
            if item[1] == task.id
            and item[2]
            in (BufferUseRole.REDUCE_INPUT, BufferUseRole.REDUCE_OUTPUT)
        )
        group_keys = tuple(item[0] for item in group_requirements)
        if len(group_keys) != len(set(group_keys)):
            _fail(
                "LOCAL_REDUCE operands must have distinct buffer keys",
                f"projection.dags.tasks.{task.id}",
            )
        if any(key in forced_region_by_key for key in group_keys):
            _fail(
                "a buffer key cannot belong to multiple LOCAL_REDUCE groups",
                f"projection.dags.tasks.{task.id}",
            )
        core_id = placement_by_task[task.id]
        core = next(item for item in cores if item.runtime_core_id == core_id)
        profile = profile_index[core.sram_profile_ref]
        required_initiators = {
            role_initiator[role]
            for key in group_keys
            for _task_id, _access, role, _view in uses_by_key[key]
        }
        candidates = tuple(
            item
            for item in sorted(
                profile.regions,
                key=lambda candidate: (candidate.base_bytes, candidate.id),
            )
            if required_initiators.issubset(item.access)
        )
        comm_candidates = tuple(
            item for item in candidates if item.name == "comm"
        )
        if comm_candidates:
            candidates = comm_candidates
        if not candidates:
            _fail(
                "LOCAL_REDUCE has no common named SRAM region",
                f"projection.dags.tasks.{task.id}",
            )
        for key in group_keys:
            forced_region_by_key[key] = candidates[0]
            allocation_keys.append(key)
    allocation_keys.extend(
        key
        for key in uses_by_key
        if key not in forced_region_by_key
        and key not in alias_root_key_by_key
    )

    for key in allocation_keys:
        core_id, value_id = key
        tensor_slice = root_by_key[key]
        value = value_index[value_id]
        core = next(item for item in cores if item.runtime_core_id == core_id)
        profile = profile_index[core.sram_profile_ref]
        required_initiators = {
            role_initiator[role]
            for _task_id, _access, role, _view in uses_by_key[key]
        }
        candidates = tuple(
            item
            for item in sorted(
                profile.regions,
                key=lambda candidate: (candidate.base_bytes, candidate.id),
            )
            if required_initiators.issubset(item.access)
        )
        if any(
            role in communication_roles
            for _task_id, _access, role, _view in uses_by_key[key]
        ):
            comm_candidates = tuple(
                item for item in candidates if item.name == "comm"
            )
            if comm_candidates:
                candidates = comm_candidates
        region = forced_region_by_key.get(key)
        if region is None and candidates:
            region = candidates[0]
        if region is None:
            _fail(
                "core SRAM profile has no region for all required initiators",
                "ir1.fabric.sram_profiles.regions",
            )
        element_bytes = _DTYPE_BYTES.get(value.dtype)
        if element_bytes is None:
            _fail(
                "unsupported buffer dtype",
                f"projection.dags.values.{value_id}.dtype",
            )
        size_bytes = math.prod(tensor_slice.shape) * element_bytes
        cursor_key = (core_id, region.id)
        is_reduce_staging = any(
            role in (BufferUseRole.REDUCE_INPUT, BufferUseRole.REDUCE_OUTPUT)
            for _task_id, _access, role, _view in uses_by_key[key]
        )
        alignment_bytes = (
            2 if is_reduce_staging else profile.allocation_alignment_bytes
        )
        offset = _align_up(
            cursor_by_core_region.get(cursor_key, 0),
            alignment_bytes,
        )
        if offset + size_bytes > region.size_bytes:
            _fail(
                (
                    "SRAM capacity exceeded for named region "
                    f"{region.name!r}: required={offset + size_bytes}, "
                    f"available={region.size_bytes}"
                ),
                f"schedule.die_{dag.die_id}.core_{core_id}.{region.id}",
            )
        cursor_by_core_region[cursor_key] = offset + size_bytes
        accesses = uses_by_key[key]
        lifetime_start = min(
            position_by_task[task_id]
            for task_id, _access, _role, _view in accesses
        )
        lifetime_end = max(
            position_by_task[task_id]
            for task_id, _access, _role, _view in accesses
        ) + 1
        owned = any(
            access is BufferAccess.WRITE
            for _task_id, access, _role, _view in accesses
        )
        identity_key = {
            "dag_id": dag.id,
            "core_id": core_id,
            "value_id": value_id,
            "tensor_slice": tensor_slice,
        }
        binding_id = stable_artifact_id(
            "buffer_binding",
            identity_key,
            schema_version=NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
        )
        storage_id = stable_artifact_id(
            "buffer_storage",
            {
                **identity_key,
                "region_ref": region.id,
                "region_offset_bytes": offset,
                "size_bytes": size_bytes,
            },
            schema_version=NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
        )
        binding_by_key[key] = BufferBinding(
            id=binding_id,
            value_id=value_id,
            tensor_slice=tensor_slice,
            core_id=core_id,
            region_ref=region.id,
            region_offset_bytes=offset,
            size_bytes=size_bytes,
            alignment_bytes=alignment_bytes,
            banks=_banks(
                region.base_bytes + offset,
                size_bytes,
                bank_count=profile.bank_count,
                interleave_bytes=profile.bank_interleave_bytes,
            ),
            storage_id=storage_id,
            alias_of=None,
            ownership=(
                BufferOwnership.OWNED if owned else BufferOwnership.BORROWED
            ),
            lifetime_start=lifetime_start,
            lifetime_end_exclusive=lifetime_end,
            dtype=value.dtype,
            layout=value.logical_layout,
        )
    for key, source_key in alias_root_key_by_key.items():
        core_id, value_id = key
        tensor_slice = root_by_key[key]
        value = value_index[value_id]
        root = binding_by_key.get(source_key)
        if root is None:
            _fail(
                "optimizer alias root must be allocated before its output",
                f"schedule.buffer_views.{value_id}",
            )
        element_bytes = _DTYPE_BYTES[value.dtype]
        size_bytes = math.prod(tensor_slice.shape) * element_bytes
        if (
            size_bytes != root.size_bytes
            or tensor_slice.offset != root.tensor_slice.offset
            or tensor_slice.shape != root.tensor_slice.shape
        ):
            _fail(
                "optimizer alias binding must equal its state root span",
                f"schedule.buffer_views.{value_id}",
            )
        accesses = uses_by_key[key]
        lifetime_start = min(
            position_by_task[task_id]
            for task_id, _access, _role, _view in accesses
        )
        lifetime_end = max(
            position_by_task[task_id]
            for task_id, _access, _role, _view in accesses
        ) + 1
        identity_key = {
            "dag_id": dag.id,
            "core_id": core_id,
            "value_id": value_id,
            "tensor_slice": tensor_slice,
        }
        binding_by_key[key] = BufferBinding(
            id=stable_artifact_id(
                "buffer_binding",
                identity_key,
                schema_version=NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
            ),
            value_id=value_id,
            tensor_slice=tensor_slice,
            core_id=root.core_id,
            region_ref=root.region_ref,
            region_offset_bytes=root.region_offset_bytes,
            size_bytes=root.size_bytes,
            alignment_bytes=root.alignment_bytes,
            banks=root.banks,
            storage_id=root.storage_id,
            alias_of=root.id,
            ownership=BufferOwnership.ALIASED,
            lifetime_start=lifetime_start,
            lifetime_end_exclusive=lifetime_end,
            dtype=value.dtype,
            layout=value.logical_layout,
        )
    buffer_bindings = tuple(binding_by_key.values())
    task_buffer_uses = tuple(
        sorted(
            (
                TaskBufferUse(
                    task_id,
                    binding_by_key[key].id,
                    access,
                    role,
                    operand_index,
                    contribution_rank,
                    tensor_slice,
                )
                for (
                    key,
                    task_id,
                    role,
                    operand_index,
                    contribution_rank,
                    access,
                    tensor_slice,
                ) in requirement_keys
            ),
            key=lambda use: (
                use.task_id,
                _USE_ROLE_ORDER[use.role],
                use.operand_index,
                (
                    use.contribution_rank
                    if use.contribution_rank is not None
                    else -1
                ),
                use.binding_id,
                use.access.value,
                use.tensor_slice.offset,
                use.tensor_slice.shape,
            ),
        )
    )
    state_tasks = tuple(
        task
        for task in topological
        if task.kind
        in (SemanticTaskKind.DMA_IN, SemanticTaskKind.DMA_OUT)
    )
    manifest = ir1.persistent_state_manifest
    if state_tasks and manifest is None:
        _fail(
            "state DMA requires a persistent-state manifest",
            "ir1.persistent_state_manifest",
        )
    hbm_by_state = (
        {
            binding.state_ref: binding
            for binding in manifest.bindings
        }
        if manifest is not None
        else {}
    )
    task_state_uses = tuple(
        sorted(
            (
                TaskStateUse(
                    task.id,
                    hbm_by_state[task.dma.state_ref].id,
                    (
                        StateUseAccess.READ
                        if task.kind is SemanticTaskKind.DMA_IN
                        else StateUseAccess.WRITE
                    ),
                )
                for task in state_tasks
                if task.dma is not None
            ),
            key=lambda use: (
                use.task_id,
                use.hbm_binding_ref,
                use.access.value,
            ),
        )
    )
    route_catalog = {
        route.id: route
        for group in ir1.groups
        for route in group.embedding.routes
    }
    for route in ir1.cross_routes:
        if route.id in route_catalog:
            _fail("route ids must be globally unique", "ir1.cross_routes")
        route_catalog[route.id] = route
    ports = {port.id: port for port in die.ports}
    cores_by_id = {core.runtime_core_id: core for core in cores}
    flow_routes: list[FlowRouteBinding] = []
    flow_by_id = {flow.id: flow for flow in dag.flows}
    for flow in dag.flows:
        route = route_catalog.get(flow.pair_route_ref)
        if route is None:
            _fail(
                "SemanticFlow references an unknown route",
                f"dag.flows.{flow.id}.pair_route_ref",
            )
        die_position = route.die_path.index(dag.die_id)
        local_task = next(
            task_by_id[task_id] for task_id in flow.task_ids
        )
        if die_position == 0:
            role = FlowRouteRole.SOURCE
            ingress = None
            first_hop = route.hops[0]
            egress = PortLeg(first_hop.link_ref, first_hop.source_port_ref)
            start = cores_by_id[placement_by_task[local_task.id]].noc_coord
            end = ports[egress.port_ref].noc_coord
        elif die_position == len(route.die_path) - 1:
            role = FlowRouteRole.DESTINATION
            last_hop = route.hops[-1]
            ingress = PortLeg(
                last_hop.link_ref,
                last_hop.destination_port_ref,
            )
            egress = None
            start = ports[ingress.port_ref].noc_coord
            end = cores_by_id[placement_by_task[local_task.id]].noc_coord
        else:
            role = FlowRouteRole.TRANSIT
            ingress_hop = route.hops[die_position - 1]
            egress_hop = route.hops[die_position]
            ingress = PortLeg(
                ingress_hop.link_ref,
                ingress_hop.destination_port_ref,
            )
            egress = PortLeg(
                egress_hop.link_ref,
                egress_hop.source_port_ref,
            )
            start = ports[ingress.port_ref].noc_coord
            end = ports[egress.port_ref].noc_coord
        flow_routes.append(
            FlowRouteBinding(
                flow.id,
                flow.pair_route_ref,
                role,
                ingress,
                egress,
                _xy_path(start, end),
            )
        )
    runtime_tasks = tuple(
        task
        for task in topological
        if task.kind
        in (
            SemanticTaskKind.SEND,
            SemanticTaskKind.RECV,
            SemanticTaskKind.WAIT,
            SemanticTaskKind.BARRIER,
        )
    )
    runtime_token_by_task = {
        task.id: stable_artifact_id(
            "runtime_token",
            {"dag_id": dag.id, "task_id": task.id},
            schema_version=NAIVE_INTRADIE_POLICY_SCHEMA_VERSION,
        )
        for task in runtime_tasks
        if task.kind is not SemanticTaskKind.WAIT
    }
    for task in runtime_tasks:
        if task.kind is not SemanticTaskKind.WAIT:
            continue
        assert task.sync is not None
        waited_recvs = tuple(
            task_by_id[dependency]
            for dependency in task.deps
            if task_by_id[dependency].kind is SemanticTaskKind.RECV
            and task_by_id[dependency].origin_ref.rank == task.origin_ref.rank
            and task_by_id[dependency].sync is not None
            and task_by_id[dependency].sync.completion_event
            == task.sync.wait_event
        )
        if len(waited_recvs) != 1:
            _fail(
                "WAIT must identify exactly one same-rank RECV dependency by wait event",
                f"projection.dags.tasks.{task.id}",
            )
        runtime_token_by_task[task.id] = runtime_token_by_task[
            waited_recvs[0].id
        ]

    runtime_bindings = tuple(
        LogicalRuntimeBinding(
            task.id,
            task.flow_id,
            (
                flow_by_id[task.flow_id].logical_channel
                if task.flow_id is not None
                else None
            ),
            (
                task.sync.wait_event
                if task.kind is SemanticTaskKind.WAIT and task.sync is not None
                else task.sync.barrier.id
                if task.kind is SemanticTaskKind.BARRIER
                and task.sync is not None
                and task.sync.barrier is not None
                else task.sync.completion_event if task.sync is not None else None
            ),
            runtime_token_by_task[task.id],
        )
        for task in runtime_tasks
    )
    result = IntraDieSchedule.create(
        producer_pass="intra_die_schedule",
        dag_id=dag.id,
        die_id=dag.die_id,
        placements=placements,
        buffer_bindings=buffer_bindings,
        task_buffer_uses=task_buffer_uses,
        task_state_uses=task_state_uses,
        flow_routes=tuple(flow_routes),
        runtime_bindings=runtime_bindings,
        core_orders=core_orders,
    )
    result.validate_against(dag, ir1)
    return result


class NaiveIntraDiePolicy:
    """Sequential allocator and component-round-robin scheduler."""

    def schedule(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
    ) -> IntraDieScheduleSet:
        if type(projection) is not IR2ProjectionResult:
            _fail("must be an IR2ProjectionResult", "projection")
        if type(ir1) is not IR1:
            _fail("must be an IR1", "ir1")
        projection.validate("projection")
        ir1.validate("ir1")
        if projection.source_ir1_id != ir1.id:
            _fail("projection references a different IR-1", "projection.source_ir1_id")
        schedules = tuple(
            _ordinary_schedule(dag, ir1) for dag in projection.dags
        )
        result = IntraDieScheduleSet.create(
            producer_pass="intra_die_schedule",
            source_projection_id=projection.id,
            source_ir1_id=ir1.id,
            schedules=schedules,
        )
        result.validate_against(projection, ir1)
        return result


__all__ = [
    "NAIVE_INTRADIE_POLICY_SCHEMA_VERSION",
    "NaiveIntraDiePolicy",
]
