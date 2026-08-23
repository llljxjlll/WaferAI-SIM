"""Build the typed whole-workload semantic-to-IR2 value bridge."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionTerminalKind,
)
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_workload import MoeSwizzleWorkloadProjection
from ..schema.swizzle_moe_workload_bridge import (
    MoeSwizzleWorkloadPhysicalUse,
    MoeSwizzleWorkloadPhysicalValueSlice,
    MoeSwizzleWorkloadValueBinding,
    MoeSwizzleWorkloadValueBridge,
    MoeSwizzleWorkloadValueKind,
)


def build_moe_swizzle_workload_value_bridge(
    execution: MoeScaleExecution,
    workload: MoeSwizzleWorkloadProjection,
    replacement_projection: MoeSwizzleIr2Projection,
) -> MoeSwizzleWorkloadValueBridge:
    """Map semantic preserved/fused boundaries onto exact physical IR2 slices.

    The implementation never decodes symbolic IDs.  Grouped-M row identity is
    carried by ordered ``original_action_refs``; combine terminal fragments are
    joined through typed assignment/N-block provenance and the original down
    GEMM action for that assignment.
    """

    if type(execution) is not MoeScaleExecution:
        raise SchemaError("requires exact MoeScaleExecution", path="moe_workload_value_bridge.execution")
    if type(workload) is not MoeSwizzleWorkloadProjection:
        raise SchemaError("requires exact whole-workload projection", path="moe_workload_value_bridge.workload")
    if type(replacement_projection) is not MoeSwizzleIr2Projection:
        raise SchemaError("requires exact IR2 projection", path="moe_workload_value_bridge.replacement_projection")
    execution.validate("moe_workload_value_bridge.execution")
    workload.validate("moe_workload_value_bridge.workload")
    replacement_projection.validate("moe_workload_value_bridge.replacement_projection")
    if (
        workload.source_execution_id != execution.id
        or replacement_projection.source_execution_id != execution.id
        or workload.source_overlay_id != replacement_projection.source_overlay_id
        or workload.replacement_projection_id != replacement_projection.id
    ):
        raise SchemaError("workload value bridge lineage is not exact", path="moe_workload_value_bridge")

    actions = {item.id: item for item in execution.actions}
    tasks = {item.id: item for item in replacement_projection.tasks}
    values = {item.id: item for item in replacement_projection.values}
    replacement_actions = {
        item.id for item in workload.actions if not item.preserved
    }
    if replacement_actions != set(tasks):
        raise SchemaError("workload replacement/task coverage is not exact", path="moe_workload_value_bridge.workload")

    producers: dict[str, list[str]] = defaultdict(list)
    consumers: dict[str, list[str]] = defaultdict(list)
    for action in execution.actions:
        for ref in action.write_values:
            producers[ref].append(action.id)
        for ref in action.read_values:
            consumers[ref].append(action.id)

    accumulated: dict[str, dict[str, object]] = {}

    def add_slice(
        semantic_ref: str,
        kind: MoeSwizzleWorkloadValueKind,
        item: MoeSwizzleWorkloadPhysicalValueSlice,
    ) -> None:
        current = accumulated.get(semantic_ref)
        if current is None:
            current = {"kind": kind, "terminal_kind": None, "slices": []}
            accumulated[semantic_ref] = current
        elif current["kind"] is not kind:
            raise SchemaError("semantic value has conflicting bridge kinds", path=semantic_ref)
        current["slices"].append(item)  # type: ignore[union-attr]

    terminal_part_by_offset: dict[str, dict[int, tuple[str, str | None]]] = {}
    for value in replacement_projection.values:
        parts: dict[int, tuple[str, str | None]] = {}
        if value.terminal_ref is not None:
            parts[0] = (value.terminal_ref, None)
        for part in value.terminal_slices:
            parts[part.byte_offset] = (part.terminal_ref, part.assignment_ref)
        if parts:
            terminal_part_by_offset[value.id] = parts

    comp_tasks = tuple(
        item for item in replacement_projection.tasks
        if item.kind is SwizzleActionKind.COMP
    )
    down_by_assignment_block: dict[tuple[str, int | None], tuple[object, object]] = {}
    packing_copy_by_key = {}
    for copy in replacement_projection.tasks:
        if copy.kind is not SwizzleActionKind.LOCAL_COPY:
            continue
        role = copy.work_role.removesuffix(".pack")
        assignment_ref = copy.assignment_refs[0] if len(copy.assignment_refs) == 1 else None
        dependency = tasks.get(copy.deps[0]) if len(copy.deps) == 1 else None
        if (
            copy.work_role not in ("gate.pack", "up.pack")
            or assignment_ref is None
            or dependency is None
            or dependency.kind is not SwizzleActionKind.COMP
            or dependency.work_role != role
            or assignment_ref not in dependency.assignment_refs
            or len(copy.read_value_refs) != 1
            or len(copy.write_value_refs) != 1
            or values[copy.read_value_refs[0]].alias_source_refs
            != (dependency.write_value_refs[0],)
            or (assignment_ref, role) in packing_copy_by_key
        ):
            raise SchemaError(
                "packing LOCAL_COPY does not close its typed COMP row",
                path=f"moe_workload_value_bridge.tasks[{copy.id}]",
            )
        packing_copy_by_key[(assignment_ref, role)] = copy

    expected_staging = set()
    expected_gemm_outputs = set()
    expected_swiglu_outputs = set()

    comp_originals_by_group: dict[
        tuple[int | None, str, tuple[str, ...]], tuple[str, ...]
    ] = {}
    for task in comp_tasks:
        if not task.original_action_refs:
            continue
        key = (task.expert_index, task.work_role, task.assignment_refs)
        previous = comp_originals_by_group.get(key)
        if previous is not None and previous != task.original_action_refs:
            raise SchemaError("COMP semantic group has conflicting original actions", path="moe_workload_value_bridge.tasks")
        comp_originals_by_group[key] = task.original_action_refs

    down_column_origin: dict[str, int] = {}
    down_groups: dict[tuple[int | None, tuple[str, ...]], list[object]] = defaultdict(list)
    for task in comp_tasks:
        if task.work_role == "down":
            if task.matmul_n is None:
                raise SchemaError("down COMP lacks typed N extent", path=task.id)
            down_groups[(task.expert_index, task.assignment_refs)].append(task)
    for group in down_groups.values():
        if len(group) > 1:
            indices = tuple(sorted(item.n_block for item in group if item.n_block is not None))
            if len(indices) != len(group) or indices != tuple(range(len(group))):
                raise SchemaError("down COMP N-block indices are not a dense typed partition", path="moe_workload_value_bridge.tasks")
        cursor = 0
        for item in sorted(
            group,
            key=lambda value: -1 if value.n_block is None else value.n_block,
        ):
            down_column_origin[item.id] = cursor
            cursor += item.matmul_n

    for task in comp_tasks:
        if (
            task.matmul_m is None or task.matmul_n is None or task.matmul_k is None
            or len(task.read_value_refs) != 2 or len(task.write_value_refs) != 1
        ):
            raise SchemaError("COMP lacks exact typed operands", path=f"moe_workload_value_bridge.tasks[{task.id}]")
        original_refs = task.original_action_refs or comp_originals_by_group.get(
            (task.expert_index, task.work_role, task.assignment_refs),
            (),
        )
        originals = tuple(actions.get(ref) for ref in original_refs)
        if (
            len(originals) != task.matmul_m
            or len(task.assignment_refs) != task.matmul_m
            or any(item is None for item in originals)
            or any(
                item.kind is not MoeScaleExecutionActionKind.GEMM
                or item.role != task.work_role
                or item.expert_index != task.expert_index
                or len(item.read_values) != 2
                or len(item.write_values) != 1
                for item in originals
            )
        ):
            raise SchemaError("COMP grouped-M original action closure is not exact", path=f"moe_workload_value_bridge.tasks[{task.id}]")
        activation, weight, output = (
            values[task.read_value_refs[0]], values[task.read_value_refs[1]],
            values[task.write_value_refs[0]],
        )
        expected_shapes = (
            (task.matmul_m, task.matmul_k),
            (task.matmul_k, task.matmul_n),
            (task.matmul_m, task.matmul_n),
        )
        if (
            (activation.shape, weight.shape, output.shape) != expected_shapes
            or any(item.dtype is not DType.FP16 for item in (activation, weight, output))
        ):
            raise SchemaError("COMP physical operand shape/dtype drifted", path=f"moe_workload_value_bridge.tasks[{task.id}]")

        column_origin = down_column_origin.get(task.id, 0)
        if task.work_role == "down":
            for row, assignment_ref in enumerate(task.assignment_refs):
                key = (assignment_ref, task.n_block)
                if key in down_by_assignment_block:
                    raise SchemaError("assignment/N-block has multiple down COMP anchors", path="moe_workload_value_bridge.tasks")
                down_by_assignment_block[key] = (task, originals[row])

        for row, original in enumerate(originals):
            assert original is not None
            staging_ref = original.read_values[1]
            staging_producers = tuple(actions[ref] for ref in producers.get(staging_ref, ()))
            if (
                len(staging_producers) != 1
                or staging_producers[0].kind is not MoeScaleExecutionActionKind.DMA_IN
                or staging_producers[0].expert_index != original.expert_index
                or staging_producers[0].id not in original.deps
            ):
                raise SchemaError("GEMM weight staging lacks exact DMA/action dependency", path=staging_ref)
            expected_staging.add(staging_ref)
            add_slice(
                staging_ref,
                MoeSwizzleWorkloadValueKind.WEIGHT_STAGING,
                MoeSwizzleWorkloadPhysicalValueSlice(
                    weight.id, task.id, original.id,
                    MoeSwizzleWorkloadPhysicalUse.IR2_READ,
                    (0, column_origin), 0, weight.size_bytes, weight.shape,
                    weight.dtype, assignment_ref=task.assignment_refs[row],
                ),
            )

            output_ref = original.write_values[0]
            expected_gemm_outputs.add(output_ref)
            output_offset = row * task.matmul_n * 2
            physical_output = output
            physical_task = task
            physical_offset = output_offset
            if task.work_role in ("gate", "up"):
                copy = packing_copy_by_key.get((task.assignment_refs[row], task.work_role))
                if copy is not None:
                    physical_output = values[copy.write_value_refs[0]]
                    physical_task = copy
                    physical_offset = 0

            terminal_part = terminal_part_by_offset.get(output.id, {}).get(output_offset)
            add_slice(
                output_ref,
                MoeSwizzleWorkloadValueKind.GEMM_OUTPUT,
                MoeSwizzleWorkloadPhysicalValueSlice(
                    physical_output.id, physical_task.id, original.id,
                    MoeSwizzleWorkloadPhysicalUse.IR2_WRITE,
                    (0, column_origin), physical_offset, task.matmul_n * 2,
                    (1, task.matmul_n), physical_output.dtype,
                    None if terminal_part is None else terminal_part[0],
                    task.assignment_refs[row],
                ),
            )

            if task.work_role in ("gate", "up"):
                swiglu_consumers = tuple(
                    actions[ref] for ref in consumers.get(output_ref, ())
                    if actions[ref].kind is MoeScaleExecutionActionKind.SWIGLU
                )
                if (
                    len(swiglu_consumers) != 1
                    or original.id not in swiglu_consumers[0].deps
                    or output_ref not in swiglu_consumers[0].read_values
                ):
                    raise SchemaError("gate/up output lacks exact SwiGLU boundary", path=output_ref)
            elif task.work_role == "down":
                swiglu_ref = original.read_values[0]
                swiglu_producers = tuple(
                    actions[ref] for ref in producers.get(swiglu_ref, ())
                )
                if (
                    len(swiglu_producers) != 1
                    or swiglu_producers[0].kind is not MoeScaleExecutionActionKind.SWIGLU
                    or swiglu_producers[0].id not in original.deps
                ):
                    raise SchemaError("down activation lacks exact SwiGLU dependency", path=swiglu_ref)
                physical_input = values[task.read_value_refs[0]]
                add_slice(
                    swiglu_ref, MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT,
                    MoeSwizzleWorkloadPhysicalValueSlice(
                        physical_input.id, task.id, original.id,
                        MoeSwizzleWorkloadPhysicalUse.IR2_READ,
                        (0, 0), row * task.matmul_k * 2,
                        task.matmul_k * 2, (1, task.matmul_k),
                        physical_input.dtype,
                        assignment_ref=task.assignment_refs[row],
                    ),
                )
            else:
                raise SchemaError("unsupported COMP semantic role", path=f"moe_workload_value_bridge.tasks[{task.id}].work_role")

    for task in replacement_projection.tasks:
        if task.kind is not SwizzleActionKind.SWIGLU:
            continue
        if len(task.read_value_refs) != 1 or len(task.write_value_refs) != 1:
            raise SchemaError("grouped SWIGLU operand arity drifted", path=task.id)
        input_value = values[task.read_value_refs[0]]
        output_value = values[task.write_value_refs[0]]
        originals = tuple(actions.get(ref) for ref in task.original_action_refs)
        if (
            len(originals) != len(task.assignment_refs)
            or output_value.shape != (len(originals), input_value.shape[1])
            or input_value.shape != (2 * len(originals), output_value.shape[1])
            or any(
                original is None
                or original.kind is not MoeScaleExecutionActionKind.SWIGLU
                or len(original.write_values) != 1
                for original in originals
            )
        ):
            raise SchemaError("grouped SWIGLU original/value closure drifted", path=task.id)
        for row, (original, assignment_ref) in enumerate(zip(
            originals, task.assignment_refs, strict=True,
        )):
            assert original is not None
            semantic_ref = original.write_values[0]
            expected_swiglu_outputs.add(semantic_ref)
            add_slice(
                semantic_ref, MoeSwizzleWorkloadValueKind.SWIGLU_OUTPUT,
                MoeSwizzleWorkloadPhysicalValueSlice(
                    output_value.id, task.id, original.id,
                    MoeSwizzleWorkloadPhysicalUse.IR2_WRITE,
                    (0, 0), row * output_value.shape[1] * 2,
                    output_value.shape[1] * 2, (1, output_value.shape[1]),
                    output_value.dtype, assignment_ref=assignment_ref,
                ),
            )
    execution_dma_staging = {
        ref for item in execution.actions
        if item.kind is MoeScaleExecutionActionKind.DMA_IN for ref in item.write_values
    }
    execution_gemm_outputs = {
        ref for item in execution.actions
        if item.kind is MoeScaleExecutionActionKind.GEMM for ref in item.write_values
    }
    execution_swiglu_outputs = {
        ref for item in execution.actions
        if item.kind is MoeScaleExecutionActionKind.SWIGLU for ref in item.write_values
    }
    if (
        expected_staging != execution_dma_staging
        or expected_gemm_outputs != execution_gemm_outputs
        or expected_swiglu_outputs != execution_swiglu_outputs
    ):
        raise SchemaError("preserved/GEMM semantic value coverage is incomplete", path="moe_workload_value_bridge.bindings")

    combined_terminals = tuple(
        item for item in execution.terminals
        if item.kind is MoeScaleExecutionTerminalKind.COMBINED
    )
    combined_by_token_expert = {
        (item.token_index, item.expert_index): item for item in combined_terminals
    }
    down_by_assignment = defaultdict(list)
    for (assignment_ref, _), anchor in down_by_assignment_block.items():
        if anchor not in down_by_assignment[assignment_ref]:
            down_by_assignment[assignment_ref].append(anchor)
    transport_widths = {}
    for value in replacement_projection.values:
        producer_task = tasks.get(value.producer_task_ref)
        if producer_task is None or producer_task.n_block is None:
            continue
        if value.terminal_ref is not None:
            if len(producer_task.assignment_refs) != 1:
                raise SchemaError("singular transport terminal lacks one assignment", path=value.id)
            parts = ((producer_task.assignment_refs[0], value.shape[-1]),)
        else:
            parts = tuple((item.assignment_ref, item.shape[-1]) for item in value.terminal_slices)
        for assignment_ref, width in parts:
            key = (assignment_ref, producer_task.n_block)
            previous = transport_widths.setdefault(key, width)
            if previous != width:
                raise SchemaError("transport N-block width is inconsistent", path=value.id)
    transport_column_origin = {}
    by_assignment = defaultdict(list)
    for (assignment_ref, n_block), width in transport_widths.items():
        by_assignment[assignment_ref].append((n_block, width))
    for assignment_ref, blocks in by_assignment.items():
        blocks.sort()
        if tuple(item[0] for item in blocks) != tuple(range(len(blocks))):
            raise SchemaError("transport N-block partition is not dense", path=assignment_ref)
        cursor = 0
        for n_block, width in blocks:
            transport_column_origin[(assignment_ref, n_block)] = cursor
            cursor += width
        full = down_by_assignment_block.get((assignment_ref, None))
        if full is not None and cursor != full[0].matmul_n:
            raise SchemaError("transport N-blocks do not cover full-N compute", path=assignment_ref)
    seen_ir2_terminals = set()
    for value in replacement_projection.values:
        parts = []
        if value.terminal_ref is not None:
            parts.append((value.terminal_ref, None, 0, value.size_bytes, value.shape))
        parts.extend(
            (item.terminal_ref, item.assignment_ref, item.byte_offset, item.size_bytes, item.shape)
            for item in value.terminal_slices
        )
        if not parts:
            continue
        producer_task = tasks.get(value.producer_task_ref)
        if producer_task is None or value.id not in producer_task.write_value_refs:
            raise SchemaError("terminal value lacks exact physical producer", path=value.id)
        for terminal_ref, assignment_ref, offset, size_bytes, shape in parts:
            if assignment_ref is None:
                if len(producer_task.assignment_refs) != 1:
                    raise SchemaError("singular terminal lacks singular assignment provenance", path=value.id)
                assignment_ref = producer_task.assignment_refs[0]
            anchor = down_by_assignment_block.get((assignment_ref, producer_task.n_block))
            if anchor is None:
                # Transport N-blocks may be a finer partition than the single
                # full-N down COMP.  The assignment itself is then the exact
                # typed join key; admitting more than one compute anchor would
                # make the terminal-to-COMP provenance ambiguous.
                assignment_anchors = down_by_assignment.get(assignment_ref, ())
                if len(assignment_anchors) == 1:
                    anchor = assignment_anchors[0]
            if anchor is None:
                raise SchemaError("terminal fragment lacks down assignment/N-block anchor", path=terminal_ref)
            down_task, down_action = anchor
            terminal = combined_by_token_expert.get(
                (down_action.token_index, down_action.expert_index),
            )
            if terminal is None:
                raise SchemaError("terminal fragment lacks typed execution terminal", path=terminal_ref)
            semantic_ref = terminal.value_ref
            down_output_ref = down_action.write_values[0]
            seen_ir2_terminals.add(terminal_ref)
            if semantic_ref == down_output_ref:
                current = accumulated.get(semantic_ref)
                if current is None or not any(
                    item.ir2_terminal_ref == terminal_ref
                    for item in current["slices"]  # type: ignore[union-attr]
                ):
                    raise SchemaError("local terminal is not the exact down output slice", path=terminal_ref)
                current["terminal_kind"] = MoeScaleExecutionTerminalKind.COMBINED
                continue
            column_origin = transport_column_origin.get(
                (assignment_ref, producer_task.n_block),
                down_column_origin[down_task.id],
            )
            add_slice(
                semantic_ref,
                MoeSwizzleWorkloadValueKind.COMBINED_TERMINAL,
                MoeSwizzleWorkloadPhysicalValueSlice(
                    value.id, producer_task.id, down_action.id,
                    MoeSwizzleWorkloadPhysicalUse.IR2_WRITE,
                    (0, column_origin), offset, size_bytes, shape, value.dtype,
                    terminal_ref, assignment_ref,
                ),
            )
            accumulated[semantic_ref]["terminal_kind"] = MoeScaleExecutionTerminalKind.COMBINED

    if seen_ir2_terminals != set(replacement_projection.terminal_refs):
        raise SchemaError("IR2 terminal fragments are not covered exactly", path="moe_workload_value_bridge.terminals")
    if not all(
        terminal.value_ref in accumulated
        and accumulated[terminal.value_ref]["terminal_kind"] is MoeScaleExecutionTerminalKind.COMBINED
        for terminal in combined_terminals
    ):
        raise SchemaError("combined semantic terminals are not bridged exactly", path="moe_workload_value_bridge.terminals")

    bindings = []
    for semantic_ref, current in sorted(accumulated.items()):
        slices = tuple(sorted(
            current["slices"],  # type: ignore[arg-type]
            key=lambda item: (
                item.semantic_origin, item.physical_task_ref,
                item.physical_value_ref, item.physical_byte_offset,
            ),
        ))
        for item in slices:
            physical = values.get(item.physical_value_ref)
            task = tasks.get(item.physical_task_ref)
            if (
                physical is None or task is None
                or item.physical_byte_offset + item.size_bytes > physical.size_bytes
                or physical.dtype is not item.dtype
                or (
                    item.use is MoeSwizzleWorkloadPhysicalUse.IR2_READ
                    and item.physical_value_ref not in task.read_value_refs
                )
                or (
                    item.use is MoeSwizzleWorkloadPhysicalUse.IR2_WRITE
                    and item.physical_value_ref not in task.write_value_refs
                )
                or item.anchor_execution_action_ref not in actions
            ):
                raise SchemaError("physical slice escapes its typed IR2 operand", path=semantic_ref)
        bindings.append(MoeSwizzleWorkloadValueBinding(
            semantic_ref,
            current["kind"],  # type: ignore[arg-type]
            tuple(producers.get(semantic_ref, ())),
            tuple(consumers.get(semantic_ref, ())),
            current["terminal_kind"],  # type: ignore[arg-type]
            slices,
        ))

    preserved_terminals = tuple(sorted(
        item.value_ref for item in execution.terminals
        if item.kind is MoeScaleExecutionTerminalKind.TAPE
    ))
    result = MoeSwizzleWorkloadValueBridge.create(
        source_execution_id=execution.id,
        source_overlay_id=workload.source_overlay_id,
        source_workload_projection_id=workload.id,
        source_replacement_projection_id=replacement_projection.id,
        bindings=tuple(bindings),
        preserved_terminal_refs=preserved_terminals,
    )
    if {
        item.semantic_value_ref for item in result.bindings
    } != execution_dma_staging | execution_gemm_outputs | execution_swiglu_outputs | {
        item.value_ref for item in combined_terminals
    }:
        raise SchemaError("semantic bridge value quotient is not exact", path="moe_workload_value_bridge.bindings")
    return result


def validate_moe_swizzle_workload_value_bridge_against(
    bridge: MoeSwizzleWorkloadValueBridge,
    execution: MoeScaleExecution,
    workload: MoeSwizzleWorkloadProjection,
    replacement_projection: MoeSwizzleIr2Projection,
) -> None:
    """Require the bridge to be the deterministic rebuild from typed truth."""

    bridge.validate("validate_moe_swizzle_workload_value_bridge_against.bridge")
    expected = build_moe_swizzle_workload_value_bridge(
        execution, workload, replacement_projection,
    )
    if bridge != expected:
        raise SchemaError(
            "workload value bridge is not the deterministic typed rebuild",
            path="validate_moe_swizzle_workload_value_bridge_against.bridge",
        )


__all__ = [
    "build_moe_swizzle_workload_value_bridge",
    "validate_moe_swizzle_workload_value_bridge_against",
]
