"""Production physical allocation for the W10 UNFUSED comparison branch."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import PlanBarrierEventPhase, RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir1 import IR1
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_abi import (
    SwizzleBarrierEventBinding,
    SwizzleTaskCoreBinding,
    SwizzleTaskRuntimeBinding,
)
from ..schema.swizzle_unfused import UnfusedComparisonPlan, UnfusedComparisonProjection
from ..schema.swizzle_unfused_abi import (
    UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION,
    UnfusedComparisonCoreABI,
    UnfusedComparisonStorageBinding,
)
from ..schema.swizzle_unfused_lowering import (
    UnfusedComparisonDteContract,
    UnfusedComparisonLoweredProgram,
    UnfusedComparisonLoweredTask,
    UnfusedComparisonMatmulContract,
    UnfusedComparisonOperandABI,
    UnfusedComparisonReduceContract,
)


def _symbol(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"unfused_comparison_{kind}",
        semantic,
        schema_version=UNFUSED_COMPARISON_CORE_ABI_SCHEMA_VERSION,
    )


def _allocate_storage_intervals(
    *,
    rank: int,
    logical_core: LogicalCoreRef,
    region_ref: str,
    region_base: int,
    region_size: int,
    storage_specs: dict[str, tuple[int, int, int]],
    contiguous_pairs: tuple[tuple[str, str], ...] = (),
    reuse_lifetimes: bool = True,
) -> tuple[UnfusedComparisonStorageBinding, ...]:
    """Place typed storage, enabling interval reuse only when explicitly requested."""

    paired_refs = {ref for pair in contiguous_pairs for ref in pair}
    if len(paired_refs) != 2 * len(contiguous_pairs):
        raise SchemaError(
            "one UNFUSED storage may belong to only one contiguous pair",
            path=f"projection.ranks[{rank}]",
        )
    units = []
    for first_ref, second_ref in contiguous_pairs:
        if first_ref not in storage_specs or second_ref not in storage_specs:
            raise SchemaError(
                "contiguous UNFUSED pair must reference exact typed storage",
                path=f"projection.ranks[{rank}]",
            )
        first = storage_specs[first_ref]
        second = storage_specs[second_ref]
        if first[0] != second[0]:
            raise SchemaError(
                "contiguous UNFUSED pair requires equal typed extents",
                path=f"projection.ranks[{rank}]",
            )
        units.append((
            (first_ref, second_ref),
            ((first_ref, 0, *first), (second_ref, first[0], *second)),
            first[0] + second[0],
            min(first[1], second[1]),
            max(first[2], second[2]),
        ))
    units.extend(
        ((storage_ref,), ((storage_ref, 0, *spec),), spec[0], spec[1], spec[2])
        for storage_ref, spec in storage_specs.items()
        if storage_ref not in paired_refs
    )
    placed_units = []
    result = []
    region_end = region_base + region_size
    aligned_region_start = (region_base + 63) // 64 * 64
    ordered_units = sorted(
        units,
        key=(
            (lambda item: (item[3], item[4], item[0]))
            if reuse_lifetimes
            else (lambda item: item[0])
        ),
    )
    for unit_key, members, unit_size, unit_start, unit_end in ordered_units:
        for storage_ref, _offset, size_bytes, _start, _end in members:
            if (
                size_bytes > region_size
                or aligned_region_start > region_end
                or size_bytes > region_end - aligned_region_start
            ):
                raise SchemaError(
                    "individual UNFUSED storage exceeds exact SRAM region",
                    path=f"projection.ranks[{rank}]",
                )
        active = sorted(
            (
                placed
                for placed in placed_units
                if not reuse_lifetimes or placed[2] > unit_start
            ),
            key=lambda item: (item[0], item[3]),
        )
        address = aligned_region_start
        for base_address, span_bytes, _end, _key in active:
            if address + unit_size <= base_address:
                break
            address = (
                (max(
                    address,
                    base_address + span_bytes,
                ) + 63)
                // 64
                * 64
            )
        if address > region_end or unit_size > region_end - address:
            raise SchemaError(
                "live UNFUSED storage set exceeds exact SRAM region",
                path=f"projection.ranks[{rank}]",
            )
        placed_units.append((address, unit_size, unit_end, unit_key))
        result.extend(
            UnfusedComparisonStorageBinding(
                rank,
                storage_ref,
                logical_core,
                region_ref,
                address + offset,
                size_bytes,
                64,
                lifetime_start,
                lifetime_end,
            )
            for storage_ref, offset, size_bytes, lifetime_start, lifetime_end
            in members
        )
    return tuple(result)


def allocate_unfused_comparison_core_abi(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
) -> UnfusedComparisonCoreABI:
    """Allocate exact typed storage with deterministic interval coloring."""

    projection.validate_against(ir1, plan)
    actions = {
        action.id: action
        for program in plan.rank_programs
        for action in program.actions
    }
    core_by_rank = {}
    task_bindings = []
    storage_bindings = []
    for rank_projection in projection.ranks:
        die = next(item for item in ir1.fabric.dies if item.id == rank_projection.die_id)
        core = min(die.cores, key=lambda item: item.local_core_id)
        logical_core = LogicalCoreRef(die.id, core.local_core_id)
        core_by_rank[rank_projection.rank] = logical_core
        task_bindings.extend(
            SwizzleTaskCoreBinding(
                task_ref=task_ref,
                rank=rank_projection.rank,
                logical_core=logical_core,
                core_order=order,
                runtime_core_id=core.runtime_core_id,
            )
            for order, task_ref in enumerate(rank_projection.task_refs)
        )
        profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
        region = profile.regions[0]
        order_by_task = {
            task_ref: order
            for order, task_ref in enumerate(rank_projection.task_refs)
        }
        storage_specs = {}
        for operand in projection.operands:
            if actions[operand.task_ref].rank != rank_projection.rank:
                continue
            order = order_by_task[operand.task_ref]
            spec = storage_specs.setdefault(
                operand.storage_ref,
                [operand.storage_bytes, order, order + 1],
            )
            if spec[0] != operand.storage_bytes:
                raise SchemaError(
                    "one typed storage ref has inconsistent spans",
                    path="projection.operands",
                )
            spec[1] = min(spec[1], order)
            spec[2] = max(spec[2], order + 1)

        views_by_task = {}
        for operand in projection.operands:
            if actions[operand.task_ref].rank == rank_projection.rank:
                views_by_task.setdefault(operand.task_ref, []).append(operand)
        contiguous_pairs = tuple(sorted(set(
            (views[0].storage_ref, views[1].storage_ref)
            for task_ref, task_views in views_by_task.items()
            if actions[task_ref].kind is SwizzleActionKind.REDUCE
            for views in (sorted(task_views, key=lambda item: item.ordinal),)
            if views[0].storage_ref != views[1].storage_ref
        )))

        storage_bindings.extend(_allocate_storage_intervals(
            rank=rank_projection.rank,
            logical_core=logical_core,
            region_ref=region.id,
            region_base=region.base_bytes,
            region_size=region.size_bytes,
            storage_specs={
                ref: (spec[0], spec[1], spec[2])
                for ref, spec in storage_specs.items()
            },
            contiguous_pairs=contiguous_pairs,
            reuse_lifetimes=len(projection.ranks) == 4,
        ))

    flow_by_task = {
        task_ref: flow
        for flow in projection.flows
        for task_ref in (flow.send_task_ref, flow.recv_task_ref)
    }
    recv_token = {
        flow.recv_task_ref: _symbol(
            "dte_token", {"projection": projection.id, "flow": flow.id}
        )
        for flow in projection.flows
    }
    wait_token = {}
    for action in actions.values():
        if action.kind is not SwizzleActionKind.WAIT:
            continue
        recv_deps = tuple(
            dependency
            for dependency in action.deps
            if actions[dependency].kind is SwizzleActionKind.RECV
        )
        if len(recv_deps) != 1:
            raise SchemaError("WAIT requires one exact RECV dependency", path="plan.rank_programs")
        wait_token[action.id] = recv_token[recv_deps[0]]
    runtime_bindings = []
    for action in actions.values():
        if action.kind not in (
            SwizzleActionKind.SEND,
            SwizzleActionKind.RECV,
            SwizzleActionKind.WAIT,
            SwizzleActionKind.LOCAL_COPY,
        ):
            continue
        flow = flow_by_task.get(action.id)
        role = (
            "send" if action.kind is SwizzleActionKind.SEND
            else "recv" if action.kind is SwizzleActionKind.RECV
            else None
        )
        runtime_bindings.append(
            SwizzleTaskRuntimeBinding(
                task_ref=action.id,
                flow_ref=flow.id if flow is not None else None,
                token_symbol_ref=(
                    recv_token[action.id]
                    if action.kind is SwizzleActionKind.RECV
                    else wait_token[action.id]
                    if action.kind is SwizzleActionKind.WAIT
                    else _symbol("dte_token", {"projection": projection.id, "task": action.id})
                    if action.kind is SwizzleActionKind.LOCAL_COPY
                    else None
                ),
                fsm_symbol_ref=(
                    _symbol("dte_fsm", {"projection": projection.id, "flow": flow.id})
                    if flow is not None else None
                ),
                peer_symbol_ref=(
                    _symbol(
                        "peer_core",
                        {"projection": projection.id, "flow": flow.id, "role": role},
                    )
                    if flow is not None else None
                ),
                peer_core=(
                    core_by_rank[action.peer_rank]
                    if flow is not None else None
                ),
            )
        )

    events = []
    barriers = {
        program.rank: next(
            (action for action in program.actions if action.kind is SwizzleActionKind.BARRIER),
            None,
        )
        for program in plan.rank_programs
    }
    if any(item is not None for item in barriers.values()):
        if any(item is None for item in barriers.values()):
            raise SchemaError("barrier must cover both ranks", path="plan.rank_programs")
        leader = barriers[0]
        peer = barriers[1]
        barrier_ref = _symbol("barrier", {"plan": plan.id, "stage": "complete"})
        for owner, source, destination, phase, opcode in (
            (peer, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_SET),
            (leader, peer, leader, PlanBarrierEventPhase.ARRIVE, RecordOpcode.EVENT_WAIT),
            (leader, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_SET),
            (peer, leader, peer, PlanBarrierEventPhase.RELEASE, RecordOpcode.EVENT_WAIT),
        ):
            semantic = {
                "projection": projection.id,
                "barrier": barrier_ref,
                "source": source.id,
                "destination": destination.id,
                "phase": phase,
            }
            events.append(
                SwizzleBarrierEventBinding(
                    barrier_ref,
                    owner.id,
                    source.id,
                    destination.id,
                    phase,
                    opcode,
                    _symbol("barrier_event", semantic),
                    _symbol("barrier_source_core", semantic),
                    _symbol("barrier_destination_core", semantic),
                    core_by_rank[source.rank],
                    core_by_rank[destination.rank],
                )
            )
    result = UnfusedComparisonCoreABI.create(
        source_ir1_id=ir1.id,
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        task_bindings=tuple(task_bindings),
        storage_bindings=tuple(storage_bindings),
        runtime_bindings=tuple(runtime_bindings),
        barrier_events=tuple(events),
    )
    return result


def build_unfused_comparison_operand_abi(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
) -> UnfusedComparisonOperandABI:
    """Freeze all record literals and contiguous reduction views without guessing."""

    projection.validate_against(ir1, plan)
    actions = {
        action.id: action
        for program in plan.rank_programs
        for action in program.actions
    }
    views_by_task = {}
    for operand in projection.operands:
        views_by_task.setdefault(operand.task_ref, []).append(operand)
    for values in views_by_task.values():
        values.sort(key=lambda item: item.ordinal)
    dtype_bytes = 2 if plan.problem.gemm.dtype.value == "fp16" else 4
    if plan.problem.gemm.accumulation_dtype is not DType.FP32:
        raise SchemaError(
            "UNFUSED comparison requires the problem's FP32 accumulation dtype",
            path="plan.problem.gemm.accumulation_dtype",
        )
    matmuls = []
    dtes = []
    reduces = []
    for action in actions.values():
        views = views_by_task.get(action.id, [])
        if action.kind is SwizzleActionKind.COMP:
            if len(views) != 3:
                raise SchemaError("GEMM requires two inputs and one output", path="projection.operands")
            lhs, rhs, output = views
            if (
                len(lhs.shape) != 2
                or len(rhs.shape) != 2
                or len(output.shape) != 2
                or lhs.shape[1] != rhs.shape[0]
                or output.shape != (lhs.shape[0], rhs.shape[1])
                or action.flops != 2 * lhs.shape[0] * lhs.shape[1] * rhs.shape[1]
            ):
                raise SchemaError(
                    "GEMM contract must equal exact rank-local typed views",
                    path="projection.operands",
                )
            matmuls.append(UnfusedComparisonMatmulContract(
                action.id,
                lhs.shape[0],
                lhs.shape[1],
                rhs.shape[1],
                plan.problem.gemm.dtype,
            ))
        elif action.kind in (
            SwizzleActionKind.SEND,
            SwizzleActionKind.RECV,
            SwizzleActionKind.LOCAL_COPY,
        ):
            dtes.append(UnfusedComparisonDteContract(
                action.id, action.logical_bytes, dtype_bytes * 8,
            ))
        elif action.kind is SwizzleActionKind.REDUCE:
            if len(views) != 3:
                raise SchemaError("REDUCE requires two inputs and one output", path="projection.operands")
            source, accumulator, output = views
            same_storage_inputs = (
                source.storage_ref == accumulator.storage_ref
                and source.byte_offset + source.byte_extent
                == accumulator.byte_offset
            )
            split_storage_inputs = (
                source.storage_ref != accumulator.storage_ref
                and source.byte_offset == accumulator.byte_offset == 0
                and source.storage_bytes == source.byte_extent
                and accumulator.storage_bytes == accumulator.byte_extent
                and source.byte_extent == accumulator.byte_extent
            )
            if (
                not (same_storage_inputs or split_storage_inputs)
                or output.storage_ref != accumulator.storage_ref
                or output.byte_offset != accumulator.byte_offset
                or output.byte_extent != accumulator.byte_extent
            ):
                raise SchemaError(
                    "REDUCE requires an exact contiguous-pair candidate and input1 update alias",
                    path="projection.operands",
                )
            reduces.append(UnfusedComparisonReduceContract(
                action.id,
                2,
                source.byte_extent // dtype_bytes,
                source.byte_extent,
                plan.problem.gemm.dtype,
                plan.problem.gemm.accumulation_dtype,
                accumulator.value_ref,
            ))
    result = UnfusedComparisonOperandABI.create(
        source_ir1_id=ir1.id,
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        operands=projection.operands,
        matmul_contracts=tuple(matmuls),
        dte_contracts=tuple(dtes),
        reduce_contracts=tuple(reduces),
    )
    return result


def lower_unfused_comparison_opcodes(
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
) -> UnfusedComparisonLoweredProgram:
    """Freeze the exact standard opcode quotient for every executable action."""

    if (projection.source_plan_ref, projection.problem_ref, projection.baseline_ref) != (
        plan.id, plan.problem.id, plan.baseline.id,
    ):
        raise SchemaError("lowering provenance is not exact", path="projection")
    quotient = {
        SwizzleActionKind.COMP: (RecordOpcode.SRAM_BIND, RecordOpcode.MATMUL),
        SwizzleActionKind.SEND: (RecordOpcode.DTE_SEND,),
        SwizzleActionKind.RECV: (RecordOpcode.DTE_RECV,),
        SwizzleActionKind.WAIT: (RecordOpcode.DTE_WAIT,),
        SwizzleActionKind.LOCAL_COPY: (RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT),
        SwizzleActionKind.REDUCE: (RecordOpcode.LOCAL_REDUCE,),
        SwizzleActionKind.BARRIER: (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT),
    }
    result = UnfusedComparisonLoweredProgram.create(
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        tasks=tuple(
            UnfusedComparisonLoweredTask(action.id, quotient[action.kind])
            for program in plan.rank_programs
            for action in program.actions
        ),
    )
    return result


__all__ = [
    "allocate_unfused_comparison_core_abi",
    "build_unfused_comparison_operand_abi",
    "lower_unfused_comparison_opcodes",
]
