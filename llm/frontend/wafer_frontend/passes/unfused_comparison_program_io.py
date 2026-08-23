"""Exact ProgramIo witnesses for formal UNFUSED comparison programs."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import BufferABI
from ..schema.ir2 import BufferAccess, BufferOwnership, BufferUseRole
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_plan import SwizzleValueUse
from ..schema.swizzle_unfused import UnfusedComparisonStage
from ..schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)


@dataclass(frozen=True, slots=True)
class UnfusedComparisonProgramIoUse:
    action_tuple_index: int
    use_tuple_index: int
    action: object
    access: BufferAccess
    role: BufferUseRole
    operand_index: int
    order_index: int
    replica_index: int = 0

    @property
    def use(self) -> "UnfusedComparisonProgramIoUse":
        return self

    @property
    def order_key(self) -> tuple[int, int, int]:
        return (self.order_index, self.action_tuple_index, self.use_tuple_index)


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
        (SwizzleActionKind.WAIT, SwizzleValueUse.READ): BufferUseRole.LOCAL_COPY_SOURCE,
        (SwizzleActionKind.WAIT, SwizzleValueUse.WRITE): BufferUseRole.LOCAL_COPY_DESTINATION,
        (SwizzleActionKind.BARRIER, SwizzleValueUse.READ): BufferUseRole.LOCAL_COPY_SOURCE,
        (SwizzleActionKind.BARRIER, SwizzleValueUse.WRITE): BufferUseRole.LOCAL_COPY_DESTINATION,
    }
    result = roles.get((kind, use))
    if result is None:
        raise SchemaError(
            "UNFUSED operand has no timing ProgramIo role",
            path="source.projection.operands",
        )
    return result


def unfused_comparison_semantic_uses(
    source: UnfusedComparisonStandardLinkedProgram,
    abis: dict[str, BufferABI],
) -> dict[str, tuple[UnfusedComparisonProgramIoUse, ...]]:
    """Map each typed subview use to both its alias and storage root ABI."""

    source.validate_against("source")
    aliases = {
        item.value_id: item for item in abis.values() if item.alias_of is not None
    }
    roots = {
        item.binding_id: item for item in abis.values() if item.alias_of is None
    }
    if len(aliases) + len(roots) != len(abis):
        raise SchemaError(
            "UNFUSED BufferABI root/subview identities are not unique",
            path="source.manifest.fragments",
        )
    actions = {
        action.id: action
        for program in source.plan.rank_programs
        for action in program.actions
    }
    flattened = {
        action.id: index
        for index, action in enumerate(
            action
            for program in source.plan.rank_programs
            for action in program.actions
        )
    }
    task_bindings = {
        item.task_ref: item for item in source.core_abi.task_bindings
    }
    collected: dict[str, list[UnfusedComparisonProgramIoUse]] = {
        abi_id: [] for abi_id in abis
    }
    for operand in source.projection.operands:
        action = actions[operand.task_ref]
        alias = aliases.get(operand.value_ref)
        root = roots.get(alias.alias_of) if alias is not None else None
        if alias is None or root is None:
            raise SchemaError(
                "typed UNFUSED operand lacks exact alias/root BufferABI closure",
                path="source.projection.operands",
            )
        binding = task_bindings[action.id]
        if alias.logical_core != binding.logical_core or root.logical_core != binding.logical_core:
            raise SchemaError(
                "UNFUSED operand/root and task core disagree",
                path="source.projection.operands",
            )
        access = (
            BufferAccess.READ
            if operand.use is SwizzleValueUse.READ
            else BufferAccess.WRITE
        )
        use = UnfusedComparisonProgramIoUse(
            flattened[action.id],
            operand.ordinal,
            action,
            access,
            _role(action.kind, operand.use),
            operand.ordinal,
            binding.core_order,
        )
        collected[alias.id].append(use)
        collected[root.id].append(use)

    result = {}
    for abi_id, uses in collected.items():
        if not uses:
            raise SchemaError(
                "UNFUSED BufferABI has no typed semantic use",
                path="source.manifest.fragments",
            )
        ordered = tuple(sorted(uses, key=lambda item: item.order_key))
        abi = abis[abi_id]
        if abi.alias_of is None:
            first_order = ordered[0].order_key[0]
            first = tuple(
                item for item in ordered if item.order_key[0] == first_order
            )
            expected = (
                BufferAccess.READ
                if abi.ownership is BufferOwnership.BORROWED
                else BufferAccess.WRITE
            )
            if (
                abi.ownership not in (
                    BufferOwnership.BORROWED,
                    BufferOwnership.OWNED,
                )
                or any(item.access is not expected for item in first)
            ):
                raise SchemaError(
                    "UNFUSED storage-root ownership disagrees with exact first use",
                    path="source.manifest.fragments",
                )
        elif abi.ownership is not BufferOwnership.ALIASED:
            raise SchemaError(
                "UNFUSED subview must be ALIASED",
                path="source.manifest.fragments",
            )
        result[abi_id] = ordered
    return result


def _unfused_comparison_terminal_roots(
    source: UnfusedComparisonStandardLinkedProgram,
) -> tuple[BufferABI, ...]:
    """Resolve one exact probeable OWNED terminal root per physical rank."""

    source.validate_against("source")
    aliases = {
        item.value_id: item
        for item in source.fragment.buffer_abi
        if item.alias_of is not None
    }
    roots = {
        item.binding_id: item
        for item in source.fragment.buffer_abi
        if item.alias_of is None
    }
    actions = {
        action.id: action
        for program in source.plan.rank_programs
        for action in program.actions
    }
    operands_by_task = {}
    for operand in source.projection.operands:
        operands_by_task.setdefault(operand.task_ref, []).append(operand)
    result = []
    for rank in source.projection.ranks:
        action = actions[rank.terminal_task_ref]
        operands = tuple(operands_by_task.get(action.id, ()))
        boundary = tuple(
            item for item in operands if item.use is SwizzleValueUse.WRITE
        )
        if not boundary and action.stage is UnfusedComparisonStage.COMPLETE:
            # AR completion carries the assembled boundary as a typed READ-only
            # subview; no producer is invented at the barrier.
            boundary = tuple(
                item for item in operands if item.use is SwizzleValueUse.READ
            )
        if not boundary:
            raise SchemaError(
                "every UNFUSED rank terminal requires typed boundary ownership",
                path="source.projection.ranks",
            )
        rank_roots = []
        for operand in boundary:
            alias = aliases.get(operand.value_ref)
            root = roots.get(alias.alias_of) if alias is not None else None
            if root is None or root.ownership is not BufferOwnership.OWNED:
                raise SchemaError(
                    "UNFUSED terminal boundary must resolve to one OWNED root",
                    path="source.manifest.fragments",
                )
            rank_roots.append(root)
        if len({item.id for item in rank_roots}) != 1:
            raise SchemaError(
                "every UNFUSED rank must resolve one exact terminal root",
                path="source.projection.ranks",
            )
        result.append(rank_roots[0])
    if (
        not result
        or len({item.id for item in result}) != len(source.projection.ranks)
    ):
        raise SchemaError(
            "UNFUSED comparison requires one distinct terminal root per rank",
            path="source.projection.ranks",
        )
    return tuple(result)


def unfused_comparison_terminal_value_ids(
    source: UnfusedComparisonStandardLinkedProgram,
) -> set[str]:
    """Return logical boundary value identities of exact terminal roots."""

    return {
        item.value_id for item in _unfused_comparison_terminal_roots(source)
    }


__all__ = [
    "UnfusedComparisonProgramIoUse",
    "unfused_comparison_semantic_uses",
    "unfused_comparison_terminal_abi_ids",
    "unfused_comparison_terminal_value_ids",
]


def unfused_comparison_terminal_abi_ids(
    source: UnfusedComparisonStandardLinkedProgram,
) -> set[str]:
    """Resolve terminal values to only their exact OWNED storage-root ABIs."""

    return {
        item.id for item in _unfused_comparison_terminal_roots(source)
    }
