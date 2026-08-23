"""Build the physical-root quotient for the complete MoE Swizzle overlay."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferOwnership
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe import MoeHardwareFacts
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_placement import build_moe_swizzle_workload_placement
from ..schema.swizzle_moe_state import MoeSwizzleWorkloadStateABI
from ..schema.swizzle_moe_workload_bridge import (
    MoeSwizzleWorkloadValueBridge,
    MoeSwizzleWorkloadValueKind,
)
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection
from ..schema.swizzle_moe_workload_abi import (
    MoeSwizzleWorkloadABI,
    MoeSwizzleWorkloadRootValuePlacement,
    MoeSwizzleWorkloadStorageRoot,
)


def _topological_actions(workload: MoeSwizzleWorkloadProjection) -> tuple[object, ...]:
    actions = {item.id: item for item in workload.actions}
    deps = {ref: set(item.deps) for ref, item in actions.items()}
    ready = sorted(ref for ref, values in deps.items() if not values)
    result = []
    while ready:
        ref = ready.pop(0)
        result.append(actions[ref])
        for other in sorted(deps):
            if ref in deps[other]:
                deps[other].remove(ref)
                if not deps[other] and actions[other] not in result and other not in ready:
                    ready.append(other)
                    ready.sort()
    if len(result) != len(actions):
        raise SchemaError("whole workload action DAG is cyclic", path="workload.actions")
    return tuple(result)


def build_moe_swizzle_workload_abi(
    ir1: IR1,
    workload: MoeSwizzleWorkloadProjection,
    replacement: MoeSwizzleIr2Projection,
    state_abi: MoeSwizzleWorkloadStateABI,
    value_bridge: MoeSwizzleWorkloadValueBridge,
    hardware_facts: MoeHardwareFacts,
) -> MoeSwizzleWorkloadABI:
    ir1.validate("build_moe_swizzle_workload_abi.ir1")
    workload.validate("build_moe_swizzle_workload_abi.workload")
    replacement.validate("build_moe_swizzle_workload_abi.replacement")
    state_abi.validate("build_moe_swizzle_workload_abi.state_abi")
    value_bridge.validate("build_moe_swizzle_workload_abi.value_bridge")
    hardware_facts.validate("build_moe_swizzle_workload_abi.hardware_facts")
    if (
        workload.replacement_projection_id != replacement.id
        or workload.state_abi_id != state_abi.id
        or state_abi.source_ir1_id != ir1.id
        or value_bridge.source_workload_projection_id != workload.id
        or value_bridge.source_replacement_projection_id != replacement.id
        or hardware_facts != MoeHardwareFacts.from_fabric(ir1.fabric)
    ):
        raise SchemaError("whole ABI source lineage is not exact", path="build_moe_swizzle_workload_abi")
    placement = {
        item.action_ref: item
        for item in build_moe_swizzle_workload_placement(
            ir1, workload, replacement, hardware_facts,
        )
    }
    tasks = {item.id: item for item in replacement.tasks}
    values = {item.id: item for item in replacement.values}
    buffers = {item.buffer_ref: item for item in replacement.buffers}
    workload_actions = {item.id: item for item in workload.actions}
    ordered = _topological_actions(workload)
    order = {item.id: index for index, item in enumerate(ordered)}
    end_of_program = len(ordered) + 1
    original_to_linked = {}
    for action in workload.actions:
        for ref in action.source_action_refs:
            if ref in original_to_linked:
                raise SchemaError("source action maps to multiple whole actions", path="workload.actions")
            original_to_linked[ref] = action.id
    bridged_action_refs = {
        ref for item in value_bridge.bindings
        for ref in item.producer_action_refs + item.consumer_action_refs
    }
    if any(ref not in original_to_linked for ref in bridged_action_refs):
        raise SchemaError("value bridge action is outside whole provenance", path="value_bridge.bindings")

    def slot_for(action_ref: str) -> int:
        owner = placement[action_ref]
        refs = (action_ref,) + owner.owner_witness_refs
        slots = {
            0 if tasks[ref].buffer_slot is None else tasks[ref].buffer_slot
            for ref in refs if ref in tasks
        }
        if len(slots) > 1:
            raise SchemaError("whole action witnesses span physical slots", path="workload.actions")
        return next(iter(slots), 0)

    groups = {}
    units = defaultdict(lambda: defaultdict(dict))
    swiglu_slots = {
        item.storage_unit_ref: item
        for item in workload.storage_slot_assignments
    }
    if len(swiglu_slots) != len(workload.storage_slot_assignments):
        raise SchemaError(
            "whole storage slot assignments are duplicated",
            path="workload.storage_slot_assignments",
        )
    state_actions = {
        item.execution_action_ref: item for item in state_abi.action_bindings
    }
    state_ref_by_staging = {
        item.staging_value_ref: item.state_ref
        for item in state_abi.action_bindings
    }
    state_unit_by_physical = {}
    for binding in value_bridge.bindings:
        if binding.kind is not MoeSwizzleWorkloadValueKind.WEIGHT_STAGING:
            continue
        state_ref = state_ref_by_staging.get(binding.semantic_value_ref)
        if state_ref is None:
            raise SchemaError(
                "weight bridge semantic value lacks StateABI identity",
                path="value_bridge.bindings",
            )
        for physical in binding.physical_slices:
            prior = state_unit_by_physical.setdefault(
                physical.physical_value_ref, state_ref,
            )
            if prior != state_ref:
                raise SchemaError(
                    "one physical RHS aliases multiple persistent states",
                    path="value_bridge.bindings",
                )

    def add(
        action_ref: str,
        family: str,
        value_ref: str,
        extent: int,
        *,
        slot: int | None = None,
        ownership: BufferOwnership = BufferOwnership.OWNED,
        terminal: bool = False,
        unit: object | None = None,
        lane: str = "value",
        placement_offset: int = 0,
        placement_extent: int | None = None,
        storage_unit_ref: str | None = None,
    ) -> None:
        owner = placement[action_ref]
        physical_slot = slot_for(action_ref) if slot is None else slot
        key = (owner.runtime_core_id, family, physical_slot)
        group = groups.setdefault(key, {
            "logical": owner.logical_core, "ownership": ownership,
            "terminal": terminal, "actions": set(), "placements": {},
        })
        if group["logical"] != owner.logical_core or group["ownership"] is not ownership or group["terminal"] != terminal:
            raise SchemaError("physical root key changes meaning", path="build_moe_swizzle_workload_abi.roots")
        group["actions"].add(action_ref)
        unit_key = value_ref if unit is None else unit
        prior = units[key][unit_key].get(lane, 0)
        units[key][unit_key][lane] = max(prior, extent)
        typed_value = values.get(value_ref)
        value_extent = (
            typed_value.size_bytes if placement_extent is None and typed_value is not None
            else extent if placement_extent is None else placement_extent
        )
        dtype = (
            typed_value.dtype if typed_value is not None
            else workload_actions[action_ref].dtype
        )
        layout = typed_value.layout if typed_value is not None else "row_major"
        unit_ref = storage_unit_ref or str(unit_key)
        value_placement = group["placements"].get(value_ref)
        meaning = (unit_key, lane, placement_offset, value_extent, dtype, layout, unit_ref)
        if value_placement is None:
            group["placements"][value_ref] = {
                "meaning": meaning, "actions": {action_ref},
            }
        elif value_placement["meaning"] != meaning:
            raise SchemaError(
                "one value has conflicting physical root placements",
                path="build_moe_swizzle_workload_abi.roots",
            )
        else:
            value_placement["actions"].add(action_ref)

    # Planner-owned reusable transport/compute buffers: mirror actual IR2 uses.
    for task in replacement.tasks:
        referenced_values = set(task.read_value_refs + task.write_value_refs)
        for use in task.buffer_uses:
            family = use.buffer_ref.rsplit(".", 1)[-1]
            if family not in ("dispatch_operand", "combine_output"):
                raise SchemaError("replacement buffer family is unsupported", path="replacement.tasks.buffer_uses")
            buffer = buffers[use.buffer_ref]
            members = tuple(ref for ref in buffer.value_refs if ref in referenced_values)
            if not members:
                raise SchemaError("buffer use has no physical value", path="replacement.tasks.buffer_uses")
            for ref in members:
                value = values[ref]
                add(
                    task.id, family, ref, buffer.size_bytes, slot=use.slot,
                    unit=use.buffer_ref, lane="buffer",
                    placement_offset=value.byte_offset,
                    placement_extent=value.size_bytes,
                    storage_unit_ref=use.buffer_ref,
                )

    # Persistent COMP RHS operands remain fixed packed roots.
    for task in replacement.tasks:
        if task.kind is not SwizzleActionKind.COMP:
            continue
        if len(task.read_value_refs) != 2 or len(task.write_value_refs) != 1:
            raise SchemaError("whole COMP operand arity drifted", path="replacement.tasks")
        _, rhs = task.read_value_refs
        storage_unit = state_unit_by_physical.get(rhs)
        if storage_unit is None:
            raise SchemaError(
                "COMP RHS lacks exact persistent-state storage identity",
                path="replacement.tasks",
            )
        add(
            task.id, f"state_stage.{task.work_role}", rhs,
            values[rhs].size_bytes, unit=storage_unit,
            storage_unit_ref=storage_unit,
        )

    # Packing copies read one exact row subview of the fixed gate/up GroupGEMM
    # root and write a planner-owned dispatch buffer.  Keep the source root
    # live through the copy without extending it to the preserved SwiGLU.
    for task in replacement.tasks:
        if task.kind is not SwizzleActionKind.LOCAL_COPY:
            continue
        if len(task.read_value_refs) != 1 or len(task.write_value_refs) != 1:
            raise SchemaError("packing copy operand arity drifted", path="replacement.tasks")
        source = values[task.read_value_refs[0]]
        if len(source.alias_source_refs) != 1:
            raise SchemaError("packing copy source lacks one typed row parent", path="replacement.tasks")
        parent = values[source.alias_source_refs[0]]
        producer = tasks.get(parent.producer_task_ref)
        role = task.work_role.removesuffix(".pack")
        if (
            producer is None or producer.kind is not SwizzleActionKind.COMP
            or producer.work_role != role or task.deps != (producer.id,)
        ):
            raise SchemaError("packing copy source COMP closure drifted", path="replacement.tasks")
        add(
            task.id, "swiglu_input", source.id, source.size_bytes,
            unit=producer.pipeline_index, lane=role,
        )

    # External source-token payloads are initialized by ProgramIo, not allocated.
    for task in replacement.tasks:
        if task.kind is not SwizzleActionKind.SEND:
            continue
        for ref in task.read_value_refs:
            value = values[ref]
            if value.borrowed and value.buffer_ref is None:
                add(
                    task.id, "boundary_input", ref, value.size_bytes,
                    ownership=BufferOwnership.BORROWED,
                )

    # Persistent-state DMA actions share the exact RHS root with their consumer.
    for action in workload.actions:
        if action.kind != "preserved.dma_in":
            continue
        source_ref = action.source_action_refs[0]
        state = state_actions[source_ref]
        family = f"state_stage.{state.role}"
        witnesses = tuple(ref for ref in placement[action.id].owner_witness_refs if ref in tasks)
        if not witnesses or any(tasks[ref].work_role != state.role for ref in witnesses):
            raise SchemaError("state DMA does not share its typed COMP owner", path="workload.actions")
        rhs_refs = {tasks[ref].read_value_refs[1] for ref in witnesses}
        if len(rhs_refs) != 1:
            raise SchemaError("state DMA witnesses do not share one physical RHS", path="workload.actions")
        rhs_ref = next(iter(rhs_refs))
        add(
            action.id, family, state.staging_value_ref, state.bytes,
            unit=state.state_ref, storage_unit_ref=state.state_ref,
        )

    # Tape terminals are direct preserved values.  Combined terminals are stored
    # in the exact contiguous packet layout written by each RECV; token/N-block
    # interpretation remains in the typed ValueBridge slices.
    for terminal in workload.terminals:
        if terminal.kind.value != "tape":
            continue
        add(
            terminal.producer_action_ref, "terminal_tape", terminal.value_ref, terminal.bytes,
            slot=0, terminal=True, unit=terminal.value_ref,
        )
    for task in replacement.tasks:
        for ref in task.write_value_refs:
            value = values[ref]
            if (
                (value.terminal_ref is None and not value.terminal_slices)
                or value.consumer_task_refs
            ):
                continue
            add(
                task.id, "terminal_combined", ref, value.size_bytes,
                slot=0, terminal=True, unit=ref,
                storage_unit_ref=ref,
            )

    # Extend physical-root lifetimes across preserved semantic boundaries.
    for binding in value_bridge.bindings:
        linked_refs = {
            original_to_linked[ref]
            for ref in binding.producer_action_refs + binding.consumer_action_refs
        }
        physical_owners = tuple(sorted({
            placement[item.physical_task_ref].runtime_core_id
            for item in binding.physical_slices
        }))
        if (
            binding.kind is not MoeSwizzleWorkloadValueKind.COMBINED_TERMINAL
            and len(physical_owners) != 1
        ):
            raise SchemaError(
                "value bridge lacks physical root for a cross-core semantic "
                f"binding {binding.semantic_value_ref!r}; "
                f"producer_read_owners={physical_owners!r}; "
                f"linked_owners={tuple((ref, placement[ref].runtime_core_id) for ref in sorted(linked_refs))!r}",
                path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
            )
        if binding.kind is MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT:
            reads = tuple(
                item for item in binding.physical_slices
                if item.use.value == "ir2_read"
            )
            writes = tuple(
                item for item in binding.physical_slices
                if item.use.value == "ir2_write"
            )
            if len(reads) != 1 or len(writes) != 1:
                raise SchemaError(
                    "SWIGLU semantic binding requires one typed write/read slice",
                    path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
                )
            read, write = reads[0], writes[0]
            read_value = values[read.physical_value_ref]
            write_value = values[write.physical_value_ref]
            if read.physical_byte_offset < write.physical_byte_offset:
                raise SchemaError(
                    "SWIGLU producer slice cannot be embedded in its consumer view",
                    path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
                )
            base = read.physical_byte_offset - write.physical_byte_offset
            storage_unit = write.physical_value_ref
            slot_assignment = swiglu_slots.get(storage_unit)
            if (
                slot_assignment is None
                or slot_assignment.writer_action_ref != write.physical_task_ref
                or slot_assignment.runtime_core_id != physical_owners[0]
            ):
                raise SchemaError(
                    "SWIGLU bridge lacks its exact whole storage slot assignment",
                    path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
                )
            add(
                read.physical_task_ref, "swiglu_output", read.physical_value_ref,
                read_value.size_bytes, slot=slot_assignment.slot,
                unit=storage_unit, lane="consumer",
                placement_offset=0, placement_extent=read_value.size_bytes,
                storage_unit_ref=storage_unit,
            )
            add(
                write.physical_task_ref, "swiglu_output", write.physical_value_ref,
                read_value.size_bytes, slot=slot_assignment.slot,
                unit=storage_unit, lane="consumer",
                placement_offset=base, placement_extent=write_value.size_bytes,
                storage_unit_ref=storage_unit,
            )
        for physical in binding.physical_slices:
            task = tasks[physical.physical_task_ref]
            if binding.kind is MoeSwizzleWorkloadValueKind.WEIGHT_STAGING:
                family = f"state_stage.{task.work_role}"
            elif binding.kind is MoeSwizzleWorkloadValueKind.GEMM_OUTPUT:
                family = (
                    "dispatch_operand" if task.work_role in ("gate", "up")
                    else "terminal_combined"
                )
            elif binding.kind is MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT:
                family = "swiglu_output"
            elif binding.kind is MoeSwizzleWorkloadValueKind.COMBINED_TERMINAL:
                family = "terminal_combined"
            else:
                raise SchemaError("unsupported value bridge kind", path="value_bridge.bindings")
            key = (
                placement[task.id].runtime_core_id,
                family,
                0 if family.startswith("terminal_") else (
                    swiglu_slots[write.physical_value_ref].slot
                    if family == "swiglu_output" else slot_for(task.id)
                ),
            )
            group = groups.get(key)
            if group is None:
                if not family.startswith("terminal_"):
                    raise SchemaError(
                        f"value bridge lacks physical root {key!r} for task {task.id!r}; "
                        f"owner={placement[task.id]!r}; linked={tuple(sorted(linked_refs))!r}; "
                        f"linked_owners={tuple((ref, placement[ref].runtime_core_id) for ref in sorted(linked_refs))!r}; "
                        f"available={tuple(sorted(item for item in groups if item[1] == family))!r}",
                        path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
                    )
                continue
            group["actions"].add(task.id)
            group["actions"].update(linked_refs)
            physical_placement = group["placements"].get(physical.physical_value_ref)
            if physical_placement is None:
                if family.startswith("terminal_"):
                    continue
                raise SchemaError(
                    "value bridge physical slice lacks a typed root placement",
                    path=f"value_bridge.bindings[{binding.semantic_value_ref}]",
                )
            physical_placement["actions"].add(task.id)
            physical_placement["actions"].update(linked_refs)

    roots = []
    for key in sorted(groups):
        runtime, family, slot = key
        group = groups[key]
        if family.startswith("terminal_"):
            extent = sum(next(iter(lanes.values())) for lanes in units[key].values())
        elif family == "swiglu_input":
            extent = max(sum(lanes.values()) for lanes in units[key].values())
        else:
            extent = max(max(lanes.values()) for lanes in units[key].values())
        action_refs = tuple(sorted(group["actions"]))
        starts = [order[ref] for ref in action_refs]
        terminal = group["terminal"]
        ownership = group["ownership"]
        unit_bases = {}
        if family.startswith("terminal_"):
            cursor = 0
            for unit_key in sorted(units[key], key=str):
                unit_bases[unit_key] = cursor
                cursor += next(iter(units[key][unit_key].values()))
        else:
            unit_bases = {unit_key: 0 for unit_key in units[key]}
        placements = []
        for value_ref, raw in group["placements"].items():
            unit_key, lane, offset, value_extent, dtype, layout, unit_ref = raw["meaning"]
            if family == "swiglu_input":
                lane_base = sum(
                    lane_extent for lane_name, lane_extent in sorted(units[key][unit_key].items())
                    if lane_name < lane
                )
            else:
                lane_base = 0
            placement_actions = tuple(sorted(raw["actions"]))
            placement_starts = [order[ref] for ref in placement_actions]
            placements.append(MoeSwizzleWorkloadRootValuePlacement(
                value_ref, unit_ref, unit_bases[unit_key] + lane_base + offset,
                value_extent, dtype, layout, min(placement_starts),
                end_of_program if terminal else max(placement_starts) + 1,
                placement_actions,
            ))
        roots.append(MoeSwizzleWorkloadStorageRoot(
            group["logical"], runtime, family, slot, extent, ownership,
            min(starts), end_of_program if terminal else max(starts) + 1,
            ownership is BufferOwnership.OWNED,
            ownership is BufferOwnership.OWNED and not terminal,
            tuple(sorted(placements, key=lambda item: (item.root_offset_bytes, item.value_ref))),
            action_refs,
        ))

    result = MoeSwizzleWorkloadABI.create(
        source_ir1_id=ir1.id,
        source_workload_projection_id=workload.id,
        source_replacement_projection_id=replacement.id,
        source_state_abi_id=state_abi.id,
        source_value_bridge_id=value_bridge.id,
        roots=tuple(roots),
    )
    return result


def validate_moe_swizzle_workload_abi_against(
    abi: MoeSwizzleWorkloadABI,
    ir1: IR1,
    workload: MoeSwizzleWorkloadProjection,
    replacement: MoeSwizzleIr2Projection,
    state_abi: MoeSwizzleWorkloadStateABI,
    value_bridge: MoeSwizzleWorkloadValueBridge,
    hardware_facts: MoeHardwareFacts,
) -> None:
    """Reject any self-consistent lifecycle/root carrier not rebuilt from truth."""

    abi.validate("validate_moe_swizzle_workload_abi_against.abi")
    expected = build_moe_swizzle_workload_abi(
        ir1, workload, replacement, state_abi, value_bridge, hardware_facts,
    )
    if abi != expected:
        raise SchemaError(
            "whole-workload ABI is not the deterministic physical-root rebuild",
            path="validate_moe_swizzle_workload_abi_against.abi",
        )


__all__ = [
    "build_moe_swizzle_workload_abi",
    "validate_moe_swizzle_workload_abi_against",
]
