"""Deterministic lowering for one planned DIRECT standalone AllGather."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import SchemaError
from ..schema.action import CollectiveAlgorithm, StandaloneCollectivePlan
from ..schema.artifact_manifest import (
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    PlanBarrierEventPhase,
    ProgramSymbol,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbol,
    RuntimeSymbolKind,
    SemanticOperandId,
    canonical_plan_barrier_core_symbol,
    canonical_plan_barrier_event_symbol,
)
from ..schema.global_action import GlobalAction, LogicalCoreRef
from ..schema.ir2 import (
    BufferBinding,
    BufferUseRole,
    FlowRouteRole,
    RegionLowering,
    SemanticTaskKind,
    StandaloneNodeOrigin,
)
from .coarse import (
    _binding_for_use,
    _buffer_abi,
    _program_symbol,
    _view_addend_for_use,
)
from .context import LoweringContext
from .isa_region import _runtime_symbol, _transport_symbols


_PRODUCER_PASS = "standalone_collective_lowering"
_RUNTIME_FIELD_ORDER = {
    field: index for index, field in enumerate(RuntimeOperandField)
}


def _unique_by_id(values: Iterable[object]) -> tuple[object, ...]:
    result: dict[str, object] = {}
    for value in values:
        value_id = getattr(value, "id")
        previous = result.setdefault(value_id, value)
        if previous != value:
            raise SchemaError("content-addressed id collision", path="lowering")
    return tuple(result[key] for key in sorted(result))


def _exact_plan_actions(
    plan: StandaloneCollectivePlan, context: LoweringContext
) -> tuple[GlobalAction, ...]:
    return tuple(
        action
        for action in context.global_dag.actions
        if isinstance(action.origin_ref, StandaloneNodeOrigin)
        and action.origin_ref.collective_plan_id == plan.id
    )


def _local_copy_records(
    action: GlobalAction,
    schedule_id: str,
    bindings: dict[str, BufferBinding],
) -> tuple[
    tuple[RelocatableRecord, ...],
    tuple[RuntimeSymbol, ...],
    tuple[ProgramSymbol, ...],
    tuple[RuntimeRelocation, ...],
    tuple[AddressRelocation, ...],
    tuple[BufferBinding, ...],
]:
    source = _binding_for_use(
        action,
        bindings,
        BufferUseRole.LOCAL_COPY_SOURCE,
        0,
        path="action.buffer_uses",
    )
    destination = _binding_for_use(
        action,
        bindings,
        BufferUseRole.LOCAL_COPY_DESTINATION,
        0,
        path="action.buffer_uses",
    )
    source_addend = _view_addend_for_use(
        action, source, BufferUseRole.LOCAL_COPY_SOURCE, 0,
        path="action.buffer_uses",
    )
    destination_addend = _view_addend_for_use(
        action, destination, BufferUseRole.LOCAL_COPY_DESTINATION, 0,
        path="action.buffer_uses",
    )
    source_address = _program_symbol(
        schedule_id=schedule_id,
        binding=source,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    destination_address = _program_symbol(
        schedule_id=schedule_id,
        binding=destination,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    token = _runtime_symbol(
        RuntimeSymbolKind.DTE_TOKEN,
        action.id,
        ("standalone_local_copy", action.id),
    )
    issue = RelocatableRecord(
        action.id,
        RecordOpcode.DTE_ISSUE,
        (
            RecordOperand.literal("direction", 0),
            RecordOperand.runtime(
                "token", RuntimeOperandField.DTE_TOKEN, token.id
            ),
            RecordOperand.literal("payload_bits", action.bytes * 8),
            RecordOperand.literal("size_bytes", action.bytes),
            RecordOperand.literal("hbm_address", 0),
            RecordOperand.address(
                "source_address",
                SemanticOperandId.SOURCE_ADDRESS,
                source_address.id,
            ),
            RecordOperand.address(
                "destination_address",
                SemanticOperandId.DESTINATION_ADDRESS,
                destination_address.id,
            ),
        ),
    )
    wait = RelocatableRecord(
        action.id,
        RecordOpcode.DTE_WAIT,
        (
            RecordOperand.runtime(
                "token", RuntimeOperandField.DTE_TOKEN, token.id
            ),
        ),
    )
    return (
        (issue, wait),
        (token,),
        (source_address, destination_address),
        (
            RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
            RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
        ),
        (
            AddressRelocation(
                0,
                SemanticOperandId.SOURCE_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                source_address.id,
                source_addend,
            ),
            AddressRelocation(
                0,
                SemanticOperandId.DESTINATION_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                destination_address.id,
                destination_addend,
            ),
        ),
        (source, destination),
    )


def _transport_record(
    action: GlobalAction,
    schedule_id: str,
    bindings: dict[str, BufferBinding],
) -> tuple[
    RelocatableRecord,
    tuple[RuntimeSymbol, ...],
    ProgramSymbol,
    tuple[RuntimeRelocation, ...],
    AddressRelocation,
    BufferBinding,
]:
    is_send = action.task_kind is SemanticTaskKind.SEND
    expected_role = FlowRouteRole.SOURCE if is_send else FlowRouteRole.DESTINATION
    if (
        action.flow is None
        or action.flow_route is None
        or action.flow_route.role is not expected_role
        or action.flow_route.flow_id != action.flow_id
        or action.bytes == 0
    ):
        raise SchemaError(
            "standalone transport must preserve its exact N3/N5 flow route and positive payload",
            path="action.flow_route",
        )
    fsm, peer = _transport_symbols(action)
    role = BufferUseRole.SEND_SOURCE if is_send else BufferUseRole.RECV_DESTINATION
    binding = _binding_for_use(
        action, bindings, role, 0, path="action.buffer_uses"
    )
    addend = _view_addend_for_use(
        action, binding, role, 0, path="action.buffer_uses"
    )
    address = _program_symbol(
        schedule_id=schedule_id,
        binding=binding,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    if is_send:
        record = RelocatableRecord(
            action.id,
            RecordOpcode.DTE_SEND,
            (
                RecordOperand.literal("mode", 0),
                RecordOperand.literal("source_space", 0),
                RecordOperand.literal("completion", 1),
                RecordOperand.literal("datatype", 0),
                RecordOperand.literal("reduce_op", 0),
                RecordOperand.runtime(
                    "fsm_id", RuntimeOperandField.DTE_FSM, fsm.id
                ),
                RecordOperand.literal("token", 0),
                RecordOperand.literal("length_bytes", action.bytes),
                RecordOperand.address(
                    "source_address",
                    SemanticOperandId.SOURCE_ADDRESS,
                    address.id,
                ),
                RecordOperand.runtime(
                    "peer_core", RuntimeOperandField.PEER_CORE, peer.id
                ),
                RecordOperand.literal("expected_sources", 0),
                RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0),
                RecordOperand.literal("collective_id", 0),
                RecordOperand.literal("epoch", 0),
            ),
        )
        operand_id = SemanticOperandId.SOURCE_ADDRESS
    else:
        record = RelocatableRecord(
            action.id,
            RecordOpcode.DTE_RECV,
            (
                RecordOperand.literal("mode", 0),
                RecordOperand.literal("completion", 1),
                RecordOperand.literal("datatype", 0),
                RecordOperand.literal("reduce_op", 0),
                RecordOperand.runtime(
                    "fsm_id", RuntimeOperandField.DTE_FSM, fsm.id
                ),
                RecordOperand.literal("token", 0),
                RecordOperand.literal("length_bytes", action.bytes),
                RecordOperand.address(
                    "destination_address",
                    SemanticOperandId.DESTINATION_ADDRESS,
                    address.id,
                ),
                RecordOperand.runtime(
                    "peer_core", RuntimeOperandField.PEER_CORE, peer.id
                ),
                RecordOperand.literal("expected_sources", 0),
                RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0),
                RecordOperand.literal("collective_id", 0),
                RecordOperand.literal("epoch", 0),
            ),
        )
        operand_id = SemanticOperandId.DESTINATION_ADDRESS
    runtime_relocations = tuple(
        sorted(
            (
                RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, peer.id),
            ),
            key=lambda item: _RUNTIME_FIELD_ORDER[item.field],
        )
    )
    return (
        record,
        (fsm, peer),
        address,
        runtime_relocations,
        AddressRelocation(
            0,
            operand_id,
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            address.id,
            addend,
        ),
        binding,
    )


def _event_record(
    *,
    dag_id: str,
    owner: GlobalAction,
    opcode: RecordOpcode,
    phase: PlanBarrierEventPhase,
    source: GlobalAction,
    destination: GlobalAction,
) -> tuple[RelocatableRecord, tuple[RuntimeSymbol, ...]]:
    source_core = canonical_plan_barrier_core_symbol(dag_id, source)
    destination_core = canonical_plan_barrier_core_symbol(dag_id, destination)
    event = canonical_plan_barrier_event_symbol(
        dag_id, phase, source, destination
    )
    operands = (
        RecordOperand.runtime(
            "source_core", RuntimeOperandField.SOURCE_CORE, source_core.id
        ),
        RecordOperand.runtime(
            "destination_core",
            RuntimeOperandField.DESTINATION_CORE,
            destination_core.id,
        ),
        RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, event.id),
    )
    if opcode is RecordOpcode.EVENT_WAIT:
        operands = (*operands, RecordOperand.literal("count", 1))
    return (
        RelocatableRecord(owner.id, opcode, operands),
        (source_core, destination_core, event),
    )


def _barrier_records(
    dag_id: str,
    action: GlobalAction,
    participants: tuple[GlobalAction, ...],
) -> tuple[tuple[RelocatableRecord, ...], tuple[RuntimeSymbol, ...]]:
    leader, *peers = participants
    emitted: list[tuple[RelocatableRecord, tuple[RuntimeSymbol, ...]]] = []
    if action.id == leader.id:
        emitted.extend(
            _event_record(
                dag_id=dag_id,
                owner=leader,
                opcode=RecordOpcode.EVENT_WAIT,
                phase=PlanBarrierEventPhase.ARRIVE,
                source=peer,
                destination=leader,
            )
            for peer in peers
        )
        emitted.extend(
            _event_record(
                dag_id=dag_id,
                owner=leader,
                opcode=RecordOpcode.EVENT_SET,
                phase=PlanBarrierEventPhase.RELEASE,
                source=leader,
                destination=peer,
            )
            for peer in peers
        )
    else:
        emitted.extend(
            (
                _event_record(
                    dag_id=dag_id,
                    owner=action,
                    opcode=RecordOpcode.EVENT_SET,
                    phase=PlanBarrierEventPhase.ARRIVE,
                    source=action,
                    destination=leader,
                ),
                _event_record(
                    dag_id=dag_id,
                    owner=action,
                    opcode=RecordOpcode.EVENT_WAIT,
                    phase=PlanBarrierEventPhase.RELEASE,
                    source=leader,
                    destination=action,
                ),
            )
        )
    return (
        tuple(record for record, _symbols in emitted),
        tuple(symbol for _record, symbols in emitted for symbol in symbols),
    )


class NaiveStandaloneCollectiveLowering:
    """Translate one already-selected DIRECT AllGather plan without replanning."""

    def __init__(self, *, validate_output: bool = True) -> None:
        self._validate_output = validate_output
        self._validated_contexts: list[LoweringContext] = []

    def _validate_context_once(self, context: LoweringContext) -> None:
        if not any(previous is context for previous in self._validated_contexts):
            context.validate()
            self._validated_contexts.append(context)

    def lower(
        self,
        actions: tuple[GlobalAction, ...],
        context: LoweringContext,
    ) -> CommandFragment:
        if type(context) is not LoweringContext:
            raise SchemaError("must be a LoweringContext", path="context")
        if (
            type(actions) is not tuple
            or not actions
            or any(type(action) is not GlobalAction for action in actions)
        ):
            raise SchemaError(
                "must be a non-empty tuple of GlobalAction", path="actions"
            )
        self._validate_context_once(context)
        plan_ids = {
            action.origin_ref.collective_plan_id
            for action in actions
            if isinstance(action.origin_ref, StandaloneNodeOrigin)
        }
        if len(plan_ids) != 1 or any(
            not isinstance(action.origin_ref, StandaloneNodeOrigin)
            for action in actions
        ):
            raise SchemaError(
                "actions must belong to one standalone collective plan",
                path="actions",
            )
        plan_id = next(iter(plan_ids))
        plan = next(
            (candidate for candidate in context.standalone_plans if candidate.id == plan_id),
            None,
        )
        if plan is None:
            raise SchemaError(
                "actions reference an unknown standalone collective plan",
                path="actions",
            )
        plan.validate_against(context.ir1, "plan")
        if plan.algorithm is not CollectiveAlgorithm.DIRECT:
            raise SchemaError(
                "standalone lowering supports only the already-planned DIRECT AllGather",
                path="plan.algorithm",
            )
        expected_actions = _exact_plan_actions(plan, context)
        if actions != expected_actions:
            raise SchemaError(
                "actions must exactly preserve the context plan action tuple including TRANSIT",
                path="actions",
            )
        allowed = {
            SemanticTaskKind.LOCAL_COPY,
            SemanticTaskKind.SEND,
            SemanticTaskKind.RECV,
            SemanticTaskKind.BARRIER,
            SemanticTaskKind.TRANSIT,
        }
        if any(
            action.task_kind not in allowed
            or action.lowering is not RegionLowering.STRICT_ACTIONS
            or (
                action.task_kind is SemanticTaskKind.TRANSIT
                and action.logical_core is not None
            )
            or (
                action.task_kind is not SemanticTaskKind.TRANSIT
                and action.logical_core is None
            )
            for action in actions
        ):
            raise SchemaError(
                "DIRECT AllGather actions must be strict local-copy/transport/barrier actions with coreless TRANSIT",
                path="actions",
            )

        executable = tuple(
            action
            for action in actions
            if action.task_kind is not SemanticTaskKind.TRANSIT
        )
        barrier_actions = tuple(
            action
            for action in executable
            if action.task_kind is SemanticTaskKind.BARRIER
        )
        if not barrier_actions:
            raise SchemaError(
                "DIRECT AllGather requires PLAN barrier actions", path="actions"
            )
        barrier = barrier_actions[0].sync.barrier
        if barrier is None or any(
            action.sync is None or action.sync.barrier != barrier
            for action in barrier_actions
        ):
            raise SchemaError(
                "all ranks must preserve one exact PLAN barrier contract",
                path="actions",
            )
        barrier_by_rank = {
            action.origin_ref.rank: action for action in barrier_actions
        }
        if set(barrier_by_rank) != set(barrier.participant_ranks):
            raise SchemaError(
                "barrier actions must exactly cover participant ranks",
                path="actions",
            )
        barrier_participants = tuple(
            barrier_by_rank[rank] for rank in barrier.participant_ranks
        )

        schedules = {
            schedule.id: schedule for schedule in context.schedule_set.schedules
        }
        actions_by_core: dict[LogicalCoreRef, list[GlobalAction]] = {}
        for action in executable:
            assert action.logical_core is not None
            actions_by_core.setdefault(action.logical_core, []).append(action)

        streams: list[CoreFragmentStream] = []
        runtime_symbols: list[RuntimeSymbol] = []
        program_symbols: list[ProgramSymbol] = []
        buffer_abis: list[BufferABI] = []
        for core in sorted(
            actions_by_core,
            key=lambda item: (item.die_id, item.local_core_id),
        ):
            core_actions = tuple(
                sorted(
                    actions_by_core[core],
                    key=lambda action: action.core_order_index,
                )
            )
            records: list[RelocatableRecord] = []
            runtime_relocations: list[RuntimeRelocation] = []
            address_relocations: list[AddressRelocation] = []
            for action in core_actions:
                schedule = schedules.get(action.source.schedule_id)
                if schedule is None or schedule.die_id != core.die_id:
                    raise SchemaError(
                        "action references an unknown die schedule",
                        path="action.source.schedule_id",
                    )
                bindings = {
                    binding.id: binding for binding in schedule.buffer_bindings
                }
                record_base = len(records)
                emitted_runtime: tuple[RuntimeRelocation, ...] = ()
                emitted_addresses: tuple[AddressRelocation, ...] = ()
                used_bindings: tuple[BufferBinding, ...] = ()
                if action.task_kind is SemanticTaskKind.LOCAL_COPY:
                    (
                        emitted,
                        action_runtime,
                        action_program,
                        emitted_runtime,
                        emitted_addresses,
                        used_bindings,
                    ) = _local_copy_records(action, schedule.id, bindings)
                elif action.task_kind in (
                    SemanticTaskKind.SEND,
                    SemanticTaskKind.RECV,
                ):
                    (
                        record,
                        action_runtime,
                        program_symbol,
                        emitted_runtime,
                        emitted_address,
                        used_binding,
                    ) = _transport_record(action, schedule.id, bindings)
                    emitted = (record,)
                    action_program = (program_symbol,)
                    emitted_addresses = (emitted_address,)
                    used_bindings = (used_binding,)
                else:
                    emitted, action_runtime = _barrier_records(
                        context.global_dag.id,
                        action,
                        barrier_participants,
                    )
                    action_program = ()
                records.extend(emitted)
                runtime_symbols.extend(action_runtime)
                program_symbols.extend(action_program)
                runtime_relocations.extend(
                    RuntimeRelocation(
                        relocation.record_index + record_base,
                        relocation.field,
                        relocation.symbol_ref,
                    )
                    for relocation in emitted_runtime
                )
                address_relocations.extend(
                    AddressRelocation(
                        relocation.record_index + record_base,
                        relocation.operand_id,
                        relocation.symbol_kind,
                        relocation.symbol_ref,
                        relocation.addend,
                    )
                    for relocation in emitted_addresses
                )
                buffer_abis.extend(
                    _buffer_abi(schedule.id, binding, core)
                    for binding in used_bindings
                )
                if action.task_kind is SemanticTaskKind.BARRIER:
                    for record_index in range(record_base, len(records)):
                        runtime_relocations.extend(
                            RuntimeRelocation(
                                record_index,
                                operand.runtime_field,
                                operand.symbol_ref,
                            )
                            for operand in records[record_index].operands
                            if operand.runtime_field is not None
                            and operand.symbol_ref is not None
                        )
            streams.append(
                CoreFragmentStream(
                    core,
                    tuple(records),
                    tuple(
                        sorted(
                            runtime_relocations,
                            key=lambda item: (
                                item.record_index,
                                _RUNTIME_FIELD_ORDER[item.field],
                            ),
                        )
                    ),
                    tuple(
                        sorted(
                            address_relocations,
                            key=lambda item: (
                                item.record_index, int(item.operand_id)
                            ),
                        )
                    ),
                )
            )

        fragment = CommandFragment.create(
            producer_pass=_PRODUCER_PASS,
            source_global_dag_id=context.global_dag.id,
            kind=FragmentKind.STANDALONE_COLLECTIVE,
            claimed_action_ids=tuple(sorted(action.id for action in executable)),
            core_streams=tuple(streams),
            runtime_symbols=_unique_by_id(runtime_symbols),
            program_symbols=_unique_by_id(program_symbols),
            buffer_abi=_unique_by_id(buffer_abis),
        )
        if self._validate_output:
            fragment.validate_against(context.global_dag)
        return fragment


__all__ = ["NaiveStandaloneCollectiveLowering"]
