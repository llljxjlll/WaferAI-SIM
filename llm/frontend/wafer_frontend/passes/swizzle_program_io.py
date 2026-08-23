"""Exact semantic-use and terminal witnesses for Swizzle timing ProgramIo."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import BufferABI
from ..schema.common import stable_artifact_id
from ..schema.ir2 import BufferAccess, BufferOwnership, BufferUseRole
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_plan import SwizzleValueUse
from ..schema.swizzle_standard import SwizzleStandardLinkedProgram
from ..schema.swizzle_ir2 import admits_wang_4rank_packed_layout


_STANDARD_LOWERING_SCHEMA = "wafer_frontend.swizzle_standard_lowering/v1alpha1"
_TERMINAL_ROOT_LAYOUT = "swizzle_standard_terminal_root/v1"
_TERMINAL_SUBVIEW_LAYOUT = "swizzle_standard_terminal_subview/v1"


@dataclass(frozen=True, slots=True)
class SwizzleProgramIoUse:
    action_tuple_index: int
    use_tuple_index: int
    action: object
    access: BufferAccess
    role: BufferUseRole
    operand_index: int
    order_index: int
    replica_index: int = 0

    @property
    def use(self) -> "SwizzleProgramIoUse":
        return self

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.order_index, self.action_tuple_index, self.use_tuple_index)


def _binding_id(source: SwizzleStandardLinkedProgram, value_ref: str, slot: int) -> str:
    return stable_artifact_id(
        "swizzle_standard_buffer_binding",
        {"core_abi": source.core_abi.id, "key": (value_ref, slot)},
        schema_version=_STANDARD_LOWERING_SCHEMA,
    )


def _role(kind: SwizzleActionKind, use: SwizzleValueUse) -> BufferUseRole:
    roles = {
        (SwizzleActionKind.COMP, SwizzleValueUse.READ): BufferUseRole.COMP_INPUT,
        (SwizzleActionKind.COMP, SwizzleValueUse.WRITE): BufferUseRole.COMP_OUTPUT,
        (SwizzleActionKind.SEND, SwizzleValueUse.READ): BufferUseRole.SEND_SOURCE,
        (SwizzleActionKind.RECV, SwizzleValueUse.WRITE): BufferUseRole.RECV_DESTINATION,
        (SwizzleActionKind.REDUCE, SwizzleValueUse.READ): BufferUseRole.REDUCE_INPUT,
        (SwizzleActionKind.REDUCE, SwizzleValueUse.WRITE): BufferUseRole.REDUCE_OUTPUT,
        (SwizzleActionKind.LOCAL_COPY, SwizzleValueUse.READ): BufferUseRole.LOCAL_COPY_SOURCE,
        (SwizzleActionKind.LOCAL_COPY, SwizzleValueUse.WRITE): BufferUseRole.LOCAL_COPY_DESTINATION,
        # WAIT is an explicit data dependency/pass-through in the W8 carrier.
        (SwizzleActionKind.WAIT, SwizzleValueUse.READ): BufferUseRole.LOCAL_COPY_SOURCE,
        (SwizzleActionKind.WAIT, SwizzleValueUse.WRITE): BufferUseRole.LOCAL_COPY_DESTINATION,
    }
    result = roles.get((kind, use))
    if result is None:
        raise SchemaError(
            "Swizzle operand has no timing ProgramIo role",
            path="source.operand_abi.operands",
        )
    return result


def swizzle_semantic_uses(
    source: SwizzleStandardLinkedProgram,
    abis: dict[str, BufferABI],
) -> dict[str, tuple[SwizzleProgramIoUse, ...]]:
    """Resolve every typed task operand to its exact standard BufferABI."""

    source.validate_against("source")
    by_binding = {abi.binding_id: abi for abi in abis.values()}
    if len(by_binding) != len(abis):
        raise SchemaError(
            "Swizzle BufferABI binding ids must be unique",
            path="source.manifest.fragments",
        )
    tasks = {
        task.id: task
        for dag in source.projection.rank_dags
        for task in dag.tasks
    }
    order = {item.task_ref: item.core_order for item in source.core_abi.task_bindings}
    flattened = {
        task.id: index
        for index, task in enumerate(
            task for dag in source.projection.rank_dags for task in dag.tasks
        )
    }
    collected: dict[str, list[SwizzleProgramIoUse]] = {
        abi_id: [] for abi_id in abis
    }
    for view in source.operand_abi.operands:
        task = tasks[view.task_ref]
        abi = by_binding.get(_binding_id(source, view.value_ref, view.slot))
        if abi is None:
            raise SchemaError(
                "typed Swizzle operand has no exact standard BufferABI",
                path="source.operand_abi.operands",
            )
        if abi.logical_core != next(
            item.logical_core
            for item in source.core_abi.task_bindings
            if item.task_ref == task.id
        ):
            raise SchemaError(
                "Swizzle operand and BufferABI core disagree",
                path="source.operand_abi.operands",
            )
        access = (
            BufferAccess.READ
            if view.use is SwizzleValueUse.READ
            else BufferAccess.WRITE
        )
        collected[abi.id].append(
            SwizzleProgramIoUse(
                flattened[task.id],
                view.ordinal,
                task,
                access,
                _role(task.kind, view.use),
                view.ordinal,
                order[task.id],
            )
        )

    result: dict[str, tuple[SwizzleProgramIoUse, ...]] = {}
    for abi_id, uses in collected.items():
        if not uses:
            abi = abis[abi_id]
            if not (
                abi.layout in (
                    _TERMINAL_ROOT_LAYOUT,
                    "swizzle_standard_storage_root/v1",
                )
                and (
                    abi.layout != _TERMINAL_ROOT_LAYOUT
                    or abi.ownership is BufferOwnership.OWNED
                )
                and abi.alias_of is None
            ):
                raise SchemaError(
                    "standard Swizzle BufferABI has no typed semantic use",
                    path="source.manifest.fragments",
                )
            result[abi_id] = ()
            continue
        ordered = tuple(sorted(uses, key=lambda item: item.order_key))
        first_order = ordered[0].order_key[0]
        first = tuple(item for item in ordered if item.order_key[0] == first_order)
        ownership = abis[abi_id].ownership
        if ownership is not BufferOwnership.ALIASED:
            expected = (
                BufferAccess.READ
                if ownership is BufferOwnership.BORROWED
                else BufferAccess.WRITE
            )
            if any(item.access is not expected for item in first):
                raise SchemaError(
                    "Swizzle BufferABI ownership disagrees with exact first use",
                    path="source.manifest.fragments",
                )
        result[abi_id] = ordered
    return result


def _swizzle_terminal_roots(
    source: SwizzleStandardLinkedProgram,
) -> tuple[BufferABI, ...]:
    abis = {item.binding_id: item for item in source.fragment.buffer_abi}
    all_roots_by_binding = {
        item.binding_id: item
        for item in source.fragment.buffer_abi
        if item.alias_of is None
    }
    roots_by_binding = {
        item.binding_id: item
        for item in source.fragment.buffer_abi
        if item.alias_of is None and item.layout == _TERMINAL_ROOT_LAYOUT
    }
    terminal_tasks = {
        ref
        for ownership in source.projection.output_ownership
        for ref in ownership.terminal_task_refs
    }
    tasks = {
        task.id: task
        for dag in source.projection.rank_dags
        for task in dag.tasks
    }
    views_by_task = {}
    for view in source.operand_abi.operands:
        views_by_task.setdefault(view.task_ref, []).append(view)
    terminal_views = []
    for task_ref in terminal_tasks:
        writes = tuple(
            view
            for view in views_by_task.get(task_ref, ())
            if view.use is SwizzleValueUse.WRITE
        )
        if not writes:
            writes = tuple(
                view
                for dependency in tasks[task_ref].deps
                for view in views_by_task.get(dependency, ())
                if view.use is SwizzleValueUse.WRITE
            )
        if not writes:
            raise SchemaError(
                "every Swizzle terminal requires a direct or immediate-dependency write",
                path="source.projection.output_ownership",
            )
        terminal_views.extend(writes)
    if not admits_wang_4rank_packed_layout(source.projection):
        legacy = {}
        for view in terminal_views:
            abi = abis.get(_binding_id(source, view.value_ref, view.slot))
            if abi is None:
                raise SchemaError(
                    "Swizzle terminal write lacks a standard BufferABI",
                    path="source.operand_abi.operands",
                )
            root = (
                all_roots_by_binding.get(abi.alias_of)
                if abi.alias_of is not None else abi
            )
            if root is None or root.ownership is not BufferOwnership.OWNED:
                raise SchemaError(
                    "legacy AR terminal must resolve to an OWNED root",
                    path="source.manifest.fragments",
                )
            legacy[root.id] = root
        return tuple(legacy[key] for key in sorted(legacy))
    result = {}
    for view in terminal_views:
        abi = abis.get(_binding_id(source, view.value_ref, view.slot))
        if abi is None:
            raise SchemaError(
                "Swizzle terminal write lacks a standard BufferABI",
                path="source.operand_abi.operands",
            )
        root = roots_by_binding.get(abi.alias_of)
        if (
            root is None
            or root.ownership is not BufferOwnership.OWNED
            or abi.layout != _TERMINAL_SUBVIEW_LAYOUT
            or abi.value_id != root.value_id
        ):
            raise SchemaError(
                "Swizzle terminal chunk must resolve to one exact OWNED root",
                path="source.manifest.fragments",
            )
        result[root.id] = root
    expected_roots = len(source.projection.output_ownership)
    if len(result) != expected_roots:
        raise SchemaError(
            "Swizzle terminals require one exact OWNED root per output owner",
            path="source.manifest.fragments",
        )
    return tuple(result[key] for key in sorted(result))


def swizzle_terminal_abi_ids(source: SwizzleStandardLinkedProgram) -> set[str]:
    """Return only the exact rank-local terminal OWNED root ABI ids."""

    source.validate_against("source")
    return {item.id for item in _swizzle_terminal_roots(source)}


def swizzle_terminal_value_ids(source: SwizzleStandardLinkedProgram) -> set[str]:
    """Return logical terminal values represented by exact rank-local roots."""

    source.validate_against("source")
    return {item.value_id for item in _swizzle_terminal_roots(source)}


__all__ = [
    "SwizzleProgramIoUse",
    "swizzle_semantic_uses",
    "swizzle_terminal_abi_ids",
    "swizzle_terminal_value_ids",
]
