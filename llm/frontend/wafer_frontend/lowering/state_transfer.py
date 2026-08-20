"""Deterministic lowering for one persistent-state transfer endpoint."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbol,
    canonical_state_transfer_wave_symbols,
    validate_state_transfer_wave_fragment_shape,
)
from ..schema.global_action import GlobalAction
from ..schema.ir2 import (
    RegionLowering,
    SemanticTaskKind,
    StateTransferOrigin,
)
from .coarse import _buffer_abi
from .context import LoweringContext
from .isa_region import (
    _RUNTIME_FIELD_ORDER,
    _token_symbol,
    _transport_record,
    _unique_by_id,
)


_PRODUCER_PASS = "state_transfer_lowering"
def _wave_record(
    dag_id: str,
    *,
    owner: GlobalAction,
    source: GlobalAction,
    destination: GlobalAction,
    opcode: RecordOpcode,
) -> tuple[RelocatableRecord, tuple[RuntimeSymbol, ...]]:
    source_core, destination_core, event = (
        canonical_state_transfer_wave_symbols(dag_id, source, destination)
    )
    if (
        opcode is RecordOpcode.EVENT_SET
        and owner.id != source.id
        or opcode is RecordOpcode.EVENT_WAIT
        and owner.id != destination.id
    ):
        raise SchemaError(
            "wave EVENT owner does not match dependency direction",
            path="actions",
        )
    operands = (
        RecordOperand.runtime(
            "source_core",
            RuntimeOperandField.SOURCE_CORE,
            source_core.id,
        ),
        RecordOperand.runtime(
            "destination_core",
            RuntimeOperandField.DESTINATION_CORE,
            destination_core.id,
        ),
        RecordOperand.runtime(
            "tag",
            RuntimeOperandField.EVENT_TAG,
            event.id,
        ),
    )
    if opcode is RecordOpcode.EVENT_WAIT:
        operands = (*operands, RecordOperand.literal("count", 1))
    return RelocatableRecord(owner.id, opcode, operands), (
        source_core,
        destination_core,
        event,
    )




class NaiveStateTransferLowering:
    """Lower exactly one ``(state-transfer contract, endpoint die)`` group."""

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

        if any(
            not isinstance(action.origin_ref, StateTransferOrigin)
            or action.lowering is not RegionLowering.STRICT_STATE_TRANSFER
            or action.logical_core is None
            or action.task_kind is SemanticTaskKind.TRANSIT
            for action in actions
        ):
            raise SchemaError(
                "state transfer fragments accept only executable STRICT_STATE_TRANSFER endpoints; TRANSIT emits no fragment",
                path="actions",
            )
        context_actions = {
            action.id: action for action in context.global_dag.actions
        }
        if any(context_actions.get(action.id) != action for action in actions):
            raise SchemaError(
                "actions must exactly equal actions in the lowering context",
                path="actions",
            )
        transfer_refs = {
            action.origin_ref.state_transfer_ref for action in actions
        }
        cores = {action.logical_core for action in actions}
        if len(transfer_refs) != 1 or len(cores) != 1:
            raise SchemaError(
                "actions must belong to one contract on one endpoint core",
                path="actions",
            )
        transfer_ref = next(iter(transfer_refs))
        core = next(iter(cores))
        assert core is not None
        expected_actions = tuple(
            action
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.origin_ref.state_transfer_ref == transfer_ref
            and action.logical_core is not None
            and action.logical_core.die_id == core.die_id
        )
        if actions != expected_actions:
            raise SchemaError(
                "actions must exactly preserve the context contract/die endpoint tuple",
                path="actions",
            )
        if actions != tuple(
            sorted(actions, key=lambda action: action.core_order_index)
        ):
            raise SchemaError(
                "endpoint actions must follow exact core order", path="actions"
            )

        kinds = tuple(action.task_kind for action in actions)
        segment_indices = tuple(
            action.origin_ref.segment_index for action in actions
        )
        legacy = all(index is None for index in segment_indices)
        is_source = (
            kinds == (SemanticTaskKind.SEND,)
            if legacy
            else all(kind is SemanticTaskKind.SEND for kind in kinds)
            and segment_indices == tuple(range(len(actions)))
        )
        is_destination = (
            kinds == (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
            if legacy
            else len(actions) % 2 == 0
            and kinds
            == tuple(
                kind
                for _ in range(len(actions) // 2)
                for kind in (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
            )
            and segment_indices
            == tuple(
                segment_index
                for segment_index in range(len(actions) // 2)
                for _ in range(2)
            )
        )
        if not (is_source or is_destination):
            raise SchemaError(
                "endpoint actions must be legacy SEND/RECV+WAIT or canonical contiguous segmented SEND/RECV+WAIT",
                path="actions",
            )
        schedule_ids = {action.source.schedule_id for action in actions}
        region_ids = {action.region_id for action in actions}
        if len(schedule_ids) != 1 or len(region_ids) != 1:
            raise SchemaError(
                "one endpoint group must preserve one schedule and region",
                path="actions",
            )
        schedule_id = next(iter(schedule_ids))
        schedule = next(
            (
                candidate
                for candidate in context.schedule_set.schedules
                if candidate.id == schedule_id
            ),
            None,
        )
        if schedule is None or schedule.die_id != core.die_id:
            raise SchemaError(
                "endpoint references an unknown die schedule",
                path="actions.source.schedule_id",
            )
        bindings = {binding.id: binding for binding in schedule.buffer_bindings}

        endpoint_units = (
            tuple((action, None) for action in actions)
            if is_source
            else tuple(zip(actions[::2], actions[1::2]))
        )
        records: list[RelocatableRecord] = []
        emitted_runtime_symbols = []
        emitted_program_symbols = []
        emitted_runtime_relocations: list[RuntimeRelocation] = []
        emitted_address_relocations: list[AddressRelocation] = []
        used_bindings = []
        dag_actions = {
            action.id: action for action in context.global_dag.actions
        }
        incoming_by_send: dict[str, GlobalAction] = {}
        outgoing_by_wait: dict[str, GlobalAction] = {}
        for destination in context.global_dag.actions:
            destination_origin = destination.origin_ref
            if (
                destination.task_kind is not SemanticTaskKind.SEND
                or not isinstance(destination_origin, StateTransferOrigin)
                or destination_origin.segment_index is None
            ):
                continue
            sources = tuple(
                dag_actions[dependency]
                for dependency in destination.deps
                if dependency in dag_actions
                and dag_actions[dependency].task_kind
                is SemanticTaskKind.WAIT
                and isinstance(
                    dag_actions[dependency].origin_ref,
                    StateTransferOrigin,
                )
                and dag_actions[dependency].origin_ref.segment_index
                is not None
                and dag_actions[dependency].logical_core
                != destination.logical_core
            )
            if len(sources) > 1:
                raise SchemaError(
                    "segmented SEND has multiple wave predecessors",
                    path="actions",
                )
            if sources:
                source = sources[0]
                incoming_by_send[destination.id] = source
                if source.id in outgoing_by_wait:
                    raise SchemaError(
                        "segmented WAIT has multiple wave successors",
                        path="actions",
                    )
                outgoing_by_wait[source.id] = destination

        validate_state_transfer_wave_fragment_shape(
            actions,
            incoming_by_send,
            outgoing_by_wait,
            "actions",
        )

        def emit_wave(
            owner: GlobalAction,
            source: GlobalAction,
            destination: GlobalAction,
            opcode: RecordOpcode,
        ) -> None:
            record, symbols = _wave_record(
                context.global_dag.id,
                owner=owner,
                source=source,
                destination=destination,
                opcode=opcode,
            )
            record_index = len(records)
            records.append(record)
            emitted_runtime_symbols.extend(symbols)
            emitted_runtime_relocations.extend(
                RuntimeRelocation(record_index, field, symbol.id)
                for field, symbol in zip(
                    (
                        RuntimeOperandField.SOURCE_CORE,
                        RuntimeOperandField.DESTINATION_CORE,
                        RuntimeOperandField.EVENT_TAG,
                    ),
                    symbols,
                )
            )
        for transport, wait in endpoint_units:
            incoming = incoming_by_send.get(transport.id)
            if incoming is not None:
                emit_wave(
                    transport,
                    incoming,
                    transport,
                    RecordOpcode.EVENT_WAIT,
                )
            token = _token_symbol(transport) if wait is not None else None
            (
                transport_record,
                runtime_symbols,
                program_symbol,
                runtime_relocations,
                address_relocation,
                used_binding,
            ) = _transport_record(
                transport,
                schedule_id,
                bindings,
                token=token,
            )
            transport_record_index = len(records)
            records.append(transport_record)
            emitted_runtime_symbols.extend(runtime_symbols)
            emitted_program_symbols.append(program_symbol)
            emitted_runtime_relocations.extend(
                RuntimeRelocation(
                    transport_record_index,
                    relocation.field,
                    relocation.symbol_ref,
                )
                for relocation in runtime_relocations
            )
            emitted_address_relocations.append(
                AddressRelocation(
                    transport_record_index,
                    address_relocation.operand_id,
                    address_relocation.symbol_kind,
                    address_relocation.symbol_ref,
                    address_relocation.addend,
                )
            )
            used_bindings.append(used_binding)
            if wait is not None:
                assert token is not None
                wait_record_index = len(records)
                records.append(
                    RelocatableRecord(
                        wait.id,
                        RecordOpcode.DTE_WAIT,
                        (
                            RecordOperand.runtime(
                                "token",
                                RuntimeOperandField.DTE_TOKEN,
                                token.id,
                            ),
                        ),
                    )
                )
                emitted_runtime_relocations.append(
                    RuntimeRelocation(
                        wait_record_index,
                        RuntimeOperandField.DTE_TOKEN,
                        token.id,
                    )
                )
                outgoing = outgoing_by_wait.get(wait.id)
                if outgoing is not None:
                    emit_wave(
                        wait,
                        wait,
                        outgoing,
                        RecordOpcode.EVENT_SET,
                    )
        unique_bindings = _unique_by_id(used_bindings)
        if len(unique_bindings) != 1:
            raise SchemaError(
                "one segmented endpoint must reuse one shared root buffer binding",
                path="actions.buffer_uses",
            )

        stream = CoreFragmentStream(
            core,
            tuple(records),
            tuple(
                sorted(
                    emitted_runtime_relocations,
                    key=lambda item: (
                        item.record_index,
                        _RUNTIME_FIELD_ORDER[item.field],
                    ),
                )
            ),
            tuple(emitted_address_relocations),
        )
        fragment = CommandFragment.create(
            producer_pass=_PRODUCER_PASS,
            source_global_dag_id=context.global_dag.id,
            kind=FragmentKind.STATE_TRANSFER,
            claimed_action_ids=tuple(sorted(action.id for action in actions)),
            core_streams=(stream,),
            runtime_symbols=_unique_by_id(emitted_runtime_symbols),
            program_symbols=_unique_by_id(emitted_program_symbols),
            buffer_abi=(
                _buffer_abi(schedule_id, unique_bindings[0], core),
            ),
            state_abi=(),
        )
        if self._validate_output:
            fragment.validate_against(context.global_dag)
        return fragment


__all__ = ["NaiveStateTransferLowering"]
