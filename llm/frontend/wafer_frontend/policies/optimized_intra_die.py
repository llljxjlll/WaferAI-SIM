"""A deterministic schedule-only intra-die optimization policy.

This is deliberately a small first O2 implementation.  It retains the N5
materializer (and therefore its complete buffer/route/runtime legality
contract), but replaces its source-order dispatch decision with a
critical-path list schedule for each already-legal executable component.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..schema.common import stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.ir2 import (
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    CoreOrder,
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieSchedule,
    IntraDieScheduleSet,
    SemanticTask,
)
from .intra_die_timing_model import estimate_task_cycles
from .naive_intra_die import NaiveIntraDiePolicy


OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION = (
    "wafer_frontend.optimized_intra_die_policy/v1"
)


@dataclass(frozen=True, slots=True)
class OptimizedIntraDieConfig:
    """Frozen O2 feature mask; defaults retain the production-safe subset."""

    schema_version: str = OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION
    critical_path_order: bool = False
    lifetime_reuse: bool = True
    bank_stagger: bool = True

    def validate(self) -> None:
        if self.schema_version != OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported optimized intra-die config schema")
        if any(type(value) is not bool for value in (self.critical_path_order, self.lifetime_reuse, self.bank_stagger)):
            raise ValueError("optimized intra-die feature mask must be bool")


def _critical_path_orders(
    dag: IntraDieDAG,
    schedule: IntraDieSchedule,
    ir1: IR1,
) -> tuple[CoreOrder, ...]:
    """Return legal deterministic per-core orders with long tails dispatched first.

    N5 requires every executable dependency to remain on one core.  Hence a
    per-core Kahn traversal is sufficient, and it gives an actual scheduling
    choice without changing projection/action/transport semantics.
    """

    task_by_id = {task.id: task for task in dag.tasks}
    source_position = {task.id: index for index, task in enumerate(dag.tasks)}
    placement = {item.task_id: item.core_id for item in schedule.placements}
    executable = set(placement)
    successors: dict[str, list[str]] = {task_id: [] for task_id in executable}
    for task in dag.tasks:
        if task.id not in executable:
            continue
        for dependency in task.deps:
            if dependency in executable:
                successors[dependency].append(task.id)

    # A byte-weighted bottom level is a cheap, deterministic proxy for the
    # critical path.  It intentionally does not use simulator feedback.
    memo: dict[str, int] = {}

    def bottom_level(task_id: str) -> int:
        cached = memo.get(task_id)
        if cached is not None:
            return cached
        task = task_by_id[task_id]
        own_cost = estimate_task_cycles(task, ir1)
        result = own_cost + max(
            (bottom_level(successor) for successor in successors[task_id]),
            default=0,
        )
        memo[task_id] = result
        return result

    orders: list[CoreOrder] = []
    for original in schedule.core_orders:
        members = set(original.task_ids)
        indegree = {
            task_id: sum(
                dependency in members
                for dependency in task_by_id[task_id].deps
            )
            for task_id in members
        }
        ready = [task_id for task_id in members if indegree[task_id] == 0]
        selected: list[str] = []
        while ready:
            # The source index and id make equal-cost choices reproducible.
            task_id = min(
                ready,
                key=lambda item: (
                    -bottom_level(item),
                    source_position[item],
                    item,
                ),
            )
            ready.remove(task_id)
            selected.append(task_id)
            for successor in successors[task_id]:
                if successor not in members:
                    continue
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        # The baseline schedule is legal, so a cycle here would be an internal
        # error.  Keep its order fail-closed rather than producing a partial one.
        if len(selected) != len(members):
            return schedule.core_orders
        orders.append(CoreOrder(original.core_id, tuple(selected)))
    return tuple(orders)


def _estimated_order_makespan(
    dag: IntraDieDAG,
    core_orders: tuple[CoreOrder, ...],
    ir1: IR1,
) -> int:
    """Evaluate dependency plus per-core issue-stream timelines."""

    tasks = {task.id: task for task in dag.tasks}
    members = {task_id for order in core_orders for task_id in order.task_ids}
    predecessors = {
        task_id: {dependency for dependency in tasks[task_id].deps if dependency in members}
        for task_id in members
    }
    for order in core_orders:
        for previous, current in zip(order.task_ids, order.task_ids[1:]):
            predecessors[current].add(previous)
    successors = {task_id: [] for task_id in members}
    indegree = {task_id: len(predecessors[task_id]) for task_id in members}
    for task_id, dependencies in predecessors.items():
        for dependency in dependencies:
            successors[dependency].append(task_id)
    finish: dict[str, int] = {}
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    while ready:
        task_id = ready.pop(0)
        start = max((finish[dependency] for dependency in predecessors[task_id]), default=0)
        finish[task_id] = start + estimate_task_cycles(tasks[task_id], ir1)
        for successor in sorted(successors[task_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(finish) != len(members):
        raise ValueError("core order plus dependencies must remain acyclic")
    return max(finish.values(), default=0)


def _with_recomputed_lifetimes(
    schedule: IntraDieSchedule,
    core_orders: tuple[CoreOrder, ...],
) -> IntraDieSchedule:
    """Rebuild the immutable schedule after changing execution positions."""

    positions = {
        task_id: index
        for order in core_orders
        for index, task_id in enumerate(order.task_ids)
    }
    uses_by_binding: dict[str, list[str]] = {
        binding.id: [] for binding in schedule.buffer_bindings
    }
    for use in schedule.task_buffer_uses:
        uses_by_binding[use.binding_id].append(use.task_id)
    bindings: list[BufferBinding] = []
    for binding in schedule.buffer_bindings:
        use_positions = [
            positions[task_id] for task_id in uses_by_binding[binding.id]
        ]
        bindings.append(
            replace(
                binding,
                lifetime_start=min(use_positions),
                lifetime_end_exclusive=max(use_positions) + 1,
            )
            if use_positions
            else binding
        )
    return IntraDieSchedule.create(
        producer_pass="intra_die_schedule",
        dag_id=schedule.dag_id,
        die_id=schedule.die_id,
        placements=schedule.placements,
        buffer_bindings=tuple(bindings),
        task_buffer_uses=schedule.task_buffer_uses,
        task_state_uses=schedule.task_state_uses,
        flow_routes=schedule.flow_routes,
        runtime_bindings=schedule.runtime_bindings,
        core_orders=core_orders,
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment

def _binding_banks(ir1: IR1, binding: BufferBinding, offset: int) -> tuple[int, ...]:
    core = next(core for die in ir1.fabric.dies for core in die.cores if core.runtime_core_id == binding.core_id)
    profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
    region = next(item for item in profile.regions if item.id == binding.region_ref)
    first = (region.base_bytes + offset) // profile.bank_interleave_bytes
    last = (region.base_bytes + offset + binding.size_bytes - 1) // profile.bank_interleave_bytes
    return tuple(sorted({index % profile.bank_count for index in range(first, last + 1)}))

def _reuse_lifetimes(schedule: IntraDieSchedule, ir1: IR1, *, bank_stagger: bool) -> IntraDieSchedule:
    transport_or_state_bindings = {
        use.binding_id
        for use in schedule.task_buffer_uses
        if use.role not in (
            BufferUseRole.COMP_INPUT,
            BufferUseRole.COMP_OUTPUT,
        )
    }
    roots = [binding for binding in schedule.buffer_bindings if binding.ownership is not BufferOwnership.ALIASED]
    aliases = [binding for binding in schedule.buffer_bindings if binding.ownership is BufferOwnership.ALIASED]
    resolved: dict[str, BufferBinding] = {}
    terminal_value_ids = {value.id for value in ir1.values if not value.consumers}
    fixed = [
        binding
        for binding in roots
        if binding.id in transport_or_state_bindings
        or binding.ownership is BufferOwnership.BORROWED
        or binding.value_id in terminal_value_ids
    ]
    for binding in fixed:
        resolved[binding.id] = binding
    ordered = sorted(
        (binding for binding in roots if binding.id not in resolved),
        key=lambda item: (item.core_id, item.region_ref, item.lifetime_start, item.lifetime_end_exclusive, -item.size_bytes, item.id),
    )
    for binding in ordered:
        core = next(core for die in ir1.fabric.dies for core in die.cores if core.runtime_core_id == binding.core_id)
        profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
        region = next(item for item in profile.regions if item.id == binding.region_ref)
        blockers = [
            item for item in (*fixed, *resolved.values())
            if item.id != binding.id
            and item.core_id == binding.core_id
            and item.region_ref == binding.region_ref
            and item.lifetime_start < binding.lifetime_end_exclusive
            and binding.lifetime_start < item.lifetime_end_exclusive
        ]
        candidates = {0}
        candidates.update(_align_up(item.region_offset_bytes + item.size_bytes, binding.alignment_bytes) for item in blockers)
        legal = []
        for offset in candidates:
            if offset + binding.size_bytes > region.size_bytes:
                continue
            if any(offset < item.region_offset_bytes + item.size_bytes and item.region_offset_bytes < offset + binding.size_bytes for item in blockers):
                continue
            banks = _binding_banks(ir1, binding, offset)
            conflict = sum(len(set(banks).intersection(item.banks)) for item in blockers)
            legal.append(((conflict if bank_stagger else 0, offset), offset, banks))
        if not legal:
            raise ValueError("optimized lifetime allocator cannot fit a baseline-valid binding")
        _score, offset, banks = min(legal, key=lambda item: item[0])
        storage_id = stable_artifact_id(
            "optimized_buffer_storage",
            {"binding_id": binding.id, "region_ref": binding.region_ref, "offset": offset, "size_bytes": binding.size_bytes},
            schema_version=OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION,
        )
        resolved[binding.id] = replace(binding, region_offset_bytes=offset, banks=banks, storage_id=storage_id)
    for alias in aliases:
        root = resolved[alias.alias_of or ""]
        resolved[alias.id] = replace(
            alias, region_offset_bytes=root.region_offset_bytes, banks=root.banks,
            storage_id=root.storage_id, core_id=root.core_id, region_ref=root.region_ref,
            size_bytes=root.size_bytes, alignment_bytes=root.alignment_bytes,
        )
    bindings = tuple(resolved[item.id] for item in schedule.buffer_bindings)
    return IntraDieSchedule.create(
        producer_pass="intra_die_schedule", dag_id=schedule.dag_id, die_id=schedule.die_id,
        placements=schedule.placements, buffer_bindings=bindings,
        task_buffer_uses=schedule.task_buffer_uses, task_state_uses=schedule.task_state_uses,
        flow_routes=schedule.flow_routes, runtime_bindings=schedule.runtime_bindings,
        core_orders=schedule.core_orders,
    )

def _reuse_double_buffer_slots(schedule: IntraDieSchedule, ir1: IR1) -> IntraDieSchedule:
    """Map versioned staging values onto two physical SRAM slots.

    Logical versions retain distinct bindings/storage IDs.  Only their byte
    offsets are shared, and only when the explicit part(i-2) dependency makes
    their lifetimes disjoint on the same core.
    """

    groups: dict[tuple[int, str], list[BufferBinding]] = {}
    for binding in schedule.buffer_bindings:
        marker = ".slot."
        version = ".version."
        if marker not in binding.value_id or version not in binding.value_id:
            continue
        prefix, suffix = binding.value_id.rsplit(version, 1)
        if not suffix.isdigit() or not prefix.rsplit(marker, 1)[-1] in ("0", "1"):
            continue
        groups.setdefault((binding.core_id, prefix), []).append(binding)

    replacements: dict[str, BufferBinding] = {}
    for members in groups.values():
        ordered = sorted(members, key=lambda item: (item.lifetime_start, item.id))
        root = ordered[0]
        previous = root
        for binding in ordered[1:]:
            compatible = (
                binding.region_ref == root.region_ref
                and binding.size_bytes == root.size_bytes
                and binding.alignment_bytes == root.alignment_bytes
                and previous.lifetime_end_exclusive <= binding.lifetime_start
            )
            if not compatible:
                raise ValueError("double-buffer slot versions must be size-compatible and lifetime-disjoint")
            replacements[binding.id] = replace(
                binding,
                region_offset_bytes=root.region_offset_bytes,
                banks=root.banks,
            )
            previous = binding
    if not replacements:
        return schedule
    return IntraDieSchedule.create(
        producer_pass="intra_die_schedule",
        dag_id=schedule.dag_id,
        die_id=schedule.die_id,
        placements=schedule.placements,
        buffer_bindings=tuple(
            replacements.get(binding.id, binding)
            for binding in schedule.buffer_bindings
        ),
        task_buffer_uses=schedule.task_buffer_uses,
        task_state_uses=schedule.task_state_uses,
        flow_routes=schedule.flow_routes,
        runtime_bindings=schedule.runtime_bindings,
        core_orders=schedule.core_orders,
    )


class OptimizedIntraDiePolicy:
    """N5-compatible critical-path dispatch policy with baseline materialization."""

    def __init__(self, config: OptimizedIntraDieConfig = OptimizedIntraDieConfig()) -> None:
        if type(config) is not OptimizedIntraDieConfig:
            raise TypeError("config must be OptimizedIntraDieConfig")
        config.validate()
        self.config = config

    def schedule(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
    ) -> IntraDieScheduleSet:
        # Delegate placement, allocation, routes and runtime bindings to the
        # proven N5 implementation; O2's decision variable in this first stage
        # is the legal execution order only.
        baseline = NaiveIntraDiePolicy().schedule(projection, ir1)
        schedules_list: list[IntraDieSchedule] = []
        for dag, schedule in zip(projection.dags, baseline.schedules, strict=True):
            selected_orders = schedule.core_orders
            if self.config.critical_path_order:
                candidate_orders = _critical_path_orders(dag, schedule, ir1)
                if _estimated_order_makespan(dag, candidate_orders, ir1) < (
                    _estimated_order_makespan(dag, schedule.core_orders, ir1)
                ):
                    selected_orders = candidate_orders
            optimized = _with_recomputed_lifetimes(schedule, selected_orders)
            refined_timing_carrier = any(
                ".split_k." in task.id for task in dag.tasks
            )
            if refined_timing_carrier:
                optimized = _reuse_double_buffer_slots(optimized, ir1)
            if self.config.lifetime_reuse and not refined_timing_carrier:
                optimized = _reuse_lifetimes(
                    optimized, ir1, bank_stagger=self.config.bank_stagger
                )
            schedules_list.append(optimized)
        schedules = tuple(schedules_list)
        result = IntraDieScheduleSet.create(
            producer_pass="intra_die_schedule",
            source_projection_id=projection.id,
            source_ir1_id=ir1.id,
            schedules=schedules,
        )
        result.validate_against(projection, ir1)
        return result


__all__ = [
    "OPTIMIZED_INTRADIE_POLICY_SCHEMA_VERSION",
    "OptimizedIntraDieConfig",
    "OptimizedIntraDiePolicy",
]
