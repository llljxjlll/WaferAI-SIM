"""Native records for one source-bound cross-replica FP32 gradient reduction."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.action import FusionActionKind
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    RecordOpcode, RecordOperand, RelocatableRecord, RuntimeOperandField,
    RuntimeRelocation,
)
from ..schema.global_action import GlobalAction
from ..schema.ir2 import RegionLowering, SemanticTaskKind, StandaloneNodeOrigin
from .coarse import _buffer_abi
from .context import LoweringContext
from .isa_region import _reduce_record, _token_symbol
from .standalone import (
    _RUNTIME_FIELD_ORDER, _local_copy_records, _transport_record,
    _unique_by_id,
)


def lower_dense_dp_gradient(
    actions: tuple[GlobalAction, ...], context: LoweringContext, gradient_index: int,
) -> CommandFragment:
    """Emit the exact N4 DP route's local half, including waited physical DTE."""

    plan = context.dp_route_plan
    projected = context.dp_projected_tasks
    dp = context.dp_replica_index
    if plan is None or projected is None or dp not in (0, 1):
        raise SchemaError("DP lowering requires a source-bound physical route", path="context.dp_route_plan")
    if gradient_index < 0 or gradient_index >= len(plan.gradients):
        raise SchemaError("unknown DP gradient", path="gradient_index")
    gradient = plan.gradients[gradient_index]
    source_actions = gradient.rank_programs[dp].actions
    expected_tasks = tuple(
        item for item in projected.tasks
        if item.replica_index == dp
        and item.state_ref == gradient.state_ref
        and item.step == gradient.step
        and item.tp_shard == gradient.tp_shard
    )
    expected_by_origin = {item.task.origin_ref.action_id: item for item in expected_tasks}
    actions_by_origin = {
        action.origin_ref.action_id: action for action in actions
        if isinstance(action.origin_ref, StandaloneNodeOrigin)
    }
    if (
        len(expected_tasks) != len(source_actions)
        or len(actions) != len(source_actions)
        or len(actions_by_origin) != len(actions)
        or set(expected_by_origin) != {item.id for item in source_actions}
        or set(actions_by_origin) != set(expected_by_origin)
    ):
        raise SchemaError("DP lowering actions must exactly cover one source gradient rank program", path="actions")
    ordered = tuple(actions_by_origin[item.id] for item in source_actions)
    if any(
        action not in context.global_dag.actions
        or action.source.task_id != expected_by_origin[source.id].task.id
        or action.origin_ref != expected_by_origin[source.id].task.origin_ref
        or action.region_id != expected_by_origin[source.id].task.region_id
        or action.lowering is not RegionLowering.STRICT_ACTIONS
        or action.task_kind is not SemanticTaskKind(source.kind.value)
        or action.logical_core is None
        or action.logical_core.die_id != expected_by_origin[source.id].die_id
        or action.bytes != expected_by_origin[source.id].task.bytes
        or action.dtype != expected_by_origin[source.id].task.dtype
        for source, action in zip(source_actions, ordered)
    ):
        raise SchemaError("DP physical action/source task/route/bytes drifted", path="actions")
    recv_by_wait: dict[str, GlobalAction] = {}
    receivers = tuple(item for item in ordered if item.task_kind is SemanticTaskKind.RECV)
    for wait in (item for item in ordered if item.task_kind is SemanticTaskKind.WAIT):
        matching = tuple(
            recv for recv in receivers
            if recv.id in wait.deps
            and recv.logical_core == wait.logical_core
            and recv.sync is not None and wait.sync is not None
            and recv.sync.completion_event == wait.sync.wait_event
            and recv.runtime_binding is not None and wait.runtime_binding is not None
            and recv.runtime_binding.token_symbol is not None
            and recv.runtime_binding.token_symbol == wait.runtime_binding.token_symbol
        )
        if len(matching) != 1 or matching[0] in recv_by_wait.values():
            raise SchemaError("DP WAIT needs exactly one same-core RECV DTE token", path="actions")
        recv_by_wait[wait.id] = matching[0]
    if set(recv_by_wait.values()) != set(receivers):
        raise SchemaError("every DP RECV needs exactly one WAIT", path="actions")

    schedules = {schedule.id: schedule for schedule in context.schedule_set.schedules}
    runtime_symbols = []
    program_symbols = []
    buffer_abis = []
    actions_by_core = {}
    for action in ordered:
        actions_by_core.setdefault(action.logical_core, []).append(action)
    streams = []
    for core in sorted(actions_by_core, key=lambda item: (item.die_id, item.local_core_id)):
        records = []
        runtime_relocations = []
        address_relocations = []
        for action in sorted(actions_by_core[core], key=lambda item: item.core_order_index):
            schedule = schedules.get(action.source.schedule_id)
            if schedule is None or schedule.die_id != core.die_id:
                raise SchemaError("DP action references another die's schedule", path="action.source.schedule_id")
            bindings = {binding.id: binding for binding in schedule.buffer_bindings}
            base = len(records)
            if action.task_kind is SemanticTaskKind.LOCAL_COPY:
                emitted, symbols, addresses, runtime_relocs, address_relocs, used = (
                    _local_copy_records(action, schedule.id, bindings)
                )
            elif action.task_kind in (SemanticTaskKind.SEND, SemanticTaskKind.RECV):
                record, symbols, address, runtime_relocs, address_reloc, used_binding = (
                    _transport_record(
                        action, schedule.id, bindings,
                        waited_recv=action.task_kind is SemanticTaskKind.RECV,
                    )
                )
                emitted, addresses, address_relocs, used = (
                    (record,), (address,), (address_reloc,), (used_binding,)
                )
            elif action.task_kind is SemanticTaskKind.WAIT:
                token = _token_symbol(recv_by_wait[action.id])
                emitted = (RelocatableRecord(
                    action.id, RecordOpcode.DTE_WAIT,
                    (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
                ),)
                symbols, addresses, address_relocs, used = (token,), (), (), ()
                runtime_relocs = (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),)
            elif action.task_kind is SemanticTaskKind.REDUCE:
                record, addresses, address_relocs, used = _reduce_record(action, schedule.id, bindings)
                emitted, symbols, runtime_relocs = (record,), (), ()
            else:
                raise SchemaError("DP route action has no native lowering", path="action.task_kind")
            records.extend(emitted)
            runtime_symbols.extend(symbols)
            program_symbols.extend(addresses)
            runtime_relocations.extend(
                RuntimeRelocation(item.record_index + base, item.field, item.symbol_ref)
                for item in runtime_relocs
            )
            address_relocations.extend(
                AddressRelocation(
                    item.record_index + base, item.operand_id, item.symbol_kind,
                    item.symbol_ref, item.addend,
                ) for item in address_relocs
            )
            buffer_abis.extend(_buffer_abi(schedule.id, binding, core) for binding in used)
        stream = CoreFragmentStream(
            core, tuple(records),
            tuple(sorted(runtime_relocations, key=lambda item: (item.record_index, _RUNTIME_FIELD_ORDER[item.field]))),
            tuple(sorted(address_relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        )
        streams.append(stream)
    return CommandFragment.create(
        producer_pass="dense_dp_gradient_lowering",
        source_global_dag_id=context.global_dag.id,
        kind=FragmentKind.STANDALONE_COLLECTIVE,
        claimed_action_ids=tuple(sorted(action.id for action in ordered)),
        core_streams=tuple(streams),
        runtime_symbols=_unique_by_id(runtime_symbols),
        program_symbols=_unique_by_id(program_symbols),
        buffer_abi=_unique_by_id(buffer_abis),
    )


__all__ = ["lower_dense_dp_gradient"]
