"""Production leaf lowering for the isolated S2-Lite DP2 rooted all-reduce."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
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
)
from ..schema.common import stable_artifact_id
from ..schema.lite_train_rooted_ar_n6 import (
    RootedArExecutableKind,
    S2LiteRootedArLoweredProgram,
    S2LiteRootedArN6Intent,
)
from ..schema.n6 import _leaf_fragment
from ..lowering.lifecycle import _core_runtime_id, _region_symbol, _storage_label_symbol
from .lower_program import _lower_fragments, _resolve_dependencies


_RUNTIME_ORDER = {field: index for index, field in enumerate(RuntimeOperandField)}


def _runtime(kind: RuntimeSymbolKind, source_ref: str, identity: object) -> RuntimeSymbol:
    semantic = {"kind": kind.value, "source_ref": source_ref, "identity": identity}
    return RuntimeSymbol(stable_artifact_id("runtime_symbol", semantic, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION), kind, source_ref)


def _address(abi, *, identity: object | None = None) -> ProgramSymbol:
    semantic = {"schedule_id": abi.schedule_id, "binding_id": abi.binding_id if identity is None else identity, "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS)}
    return ProgramSymbol(stable_artifact_id("program_symbol", semantic, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION), ProgramSymbolKind.ABSOLUTE_ADDRESS, abi.binding_id)


def _fragment(intent, unit_ids, core, records, runtime_symbols, program_symbols, runtime_relocs, address_relocs, abis):
    return CommandFragment.create(
        producer_pass="s2_lite_rooted_ar_lowering",
        source_global_dag_id=intent.source.id,
        kind=FragmentKind.S2_LITE_ROOTED_AR,
        claimed_action_ids=tuple(sorted(unit_ids)),
        core_streams=(CoreFragmentStream(
            core,
            records,
            tuple(sorted(runtime_relocs, key=lambda item: (item.record_index, _RUNTIME_ORDER[item.field]))),
            tuple(sorted(address_relocs, key=lambda item: (item.record_index, int(item.operand_id)))),
        ),),
        runtime_symbols=tuple(sorted({item.id: item for item in runtime_symbols}.values(), key=lambda item: item.id)),
        program_symbols=tuple(sorted({item.id: item for item in program_symbols}.values(), key=lambda item: item.id)),
        buffer_abi=tuple(sorted({item.id: item for item in abis}.values(), key=lambda item: item.id)),
        state_abi=(),
    )


def _dte_record(unit, address, fsm, peer, token, *, send: bool):
    token_operand = RecordOperand.literal("token", 0) if token is None else RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id)
    operands = (
        *((RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0), RecordOperand.literal("completion", 1)) if send else (RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0))),
        RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm.id), token_operand,
        RecordOperand.literal("length_bytes", unit.bytes),
        RecordOperand.address("source_address" if send else "destination_address", SemanticOperandId.SOURCE_ADDRESS if send else SemanticOperandId.DESTINATION_ADDRESS, address.id),
        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer.id),
        RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
        RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0), RecordOperand.literal("epoch", 0),
    )
    return RelocatableRecord(unit.id, RecordOpcode.DTE_SEND if send else RecordOpcode.DTE_RECV, operands)


def _alloc_record(action_id, abi, context):
    runtime_core_id = _core_runtime_id(context, abi.logical_core)
    region = _region_symbol(abi.region_ref)
    label = _storage_label_symbol(abi.schedule_id, runtime_core_id, abi.storage_id)
    return (
        RelocatableRecord(action_id, RecordOpcode.SRAM_ALLOC_AT, (
            RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.id),
            RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
            RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
            RecordOperand.literal("size_bytes", abi.size_bytes),
            RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
            RecordOperand.literal("lifetime", 0),
            RecordOperand.literal("spillable", False),
        )),
        (region, label),
    )


def _free_record(action_id, abi, context):
    label = _storage_label_symbol(abi.schedule_id, _core_runtime_id(context, abi.logical_core), abi.storage_id)
    return (
        RelocatableRecord(action_id, RecordOpcode.SRAM_FREE, (
            RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.id),
        )),
        label,
    )


def _overlay(intent: S2LiteRootedArN6Intent) -> tuple[CommandFragment, ...]:
    abi = {item.id: item for item in intent.gradient_buffer_abis + intent.scratch_buffer_abis}
    copy, us, ur, uw, reduce, ds, dr, dw = intent.units
    fragments = []
    # Root local copy into rank-0 scratch.
    source, destination = abi[copy.input_buffer_abi_refs[0]], abi[copy.output_buffer_abi_ref]
    source_symbol, destination_symbol = _address(source), _address(destination)
    copy_token = _runtime(RuntimeSymbolKind.DTE_TOKEN, copy.id, ("rooted_ar_copy", copy.id))
    issue = RelocatableRecord(copy.id, RecordOpcode.DTE_ISSUE, (
        RecordOperand.literal("direction", 0), RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, copy_token.id),
        RecordOperand.literal("payload_bits", copy.bytes * 8), RecordOperand.literal("size_bytes", copy.bytes), RecordOperand.literal("hbm_address", 0),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination_symbol.id),
    ))
    wait = RelocatableRecord(copy.id, RecordOpcode.DTE_WAIT, (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, copy_token.id),))
    alloc0, alloc0_symbols = _alloc_record(copy.id, destination, intent.lowering_contexts[0])
    fragments.append(_fragment(intent, (copy.id,), copy.logical_core, (alloc0, issue, wait), (copy_token,), (*alloc0_symbols, source_symbol, destination_symbol), (
        RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, copy_token.id), RuntimeRelocation(2, RuntimeOperandField.DTE_TOKEN, copy_token.id)), (
        AddressRelocation(0, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, alloc0_symbols[0].id, 0),
        AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, alloc0_symbols[1].id, 0),
        AddressRelocation(1, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
        AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0)), (source, destination)))

    def transfer(send_unit, recv_unit, wait_unit, source_abi, destination_abi, label, *, allocate_destination=False, free_after_send=()):
        fsm = _runtime(RuntimeSymbolKind.DTE_FSM, send_unit.source_step_ref, (label, "fsm"))
        token = _runtime(RuntimeSymbolKind.DTE_TOKEN, recv_unit.id, (label, "recv"))
        send_peer = _runtime(RuntimeSymbolKind.RUNTIME_CORE, recv_unit.id, (label, "send_peer"))
        recv_peer = _runtime(RuntimeSymbolKind.RUNTIME_CORE, send_unit.id, (label, "recv_peer"))
        ss, dsym = _address(source_abi), _address(destination_abi)
        send_record = _dte_record(send_unit, ss, fsm, send_peer, None, send=True)
        recv_record = _dte_record(recv_unit, dsym, fsm, recv_peer, token, send=False)
        wait_record = RelocatableRecord(wait_unit.id, RecordOpcode.DTE_WAIT, (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),))
        send_records = [send_record]
        send_symbols = [ss]
        send_address_relocs = [AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, ss.id, 0)]
        send_abis = [source_abi]
        for free_abi in free_after_send:
            free_record, free_symbol = _free_record(send_unit.id, free_abi, intent.lowering_contexts[0])
            record_index = len(send_records)
            send_records.append(free_record)
            send_symbols.append(free_symbol)
            send_address_relocs.append(AddressRelocation(record_index, SemanticOperandId.SYMBOL, ProgramSymbolKind.SRAM_LABEL, free_symbol.id, 0))
            send_abis.append(free_abi)
        send_fragment = _fragment(intent, (send_unit.id,), send_unit.logical_core, tuple(send_records), (fsm, send_peer), tuple(send_symbols), (
            RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id), RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, send_peer.id)), (
            *send_address_relocs,), tuple(send_abis))
        recv_records = []
        recv_symbols = [dsym]
        recv_address_relocs = []
        if allocate_destination:
            alloc, alloc_symbols = _alloc_record(recv_unit.id, destination_abi, intent.lowering_contexts[0])
            recv_records.append(alloc)
            recv_symbols.extend(alloc_symbols)
            recv_address_relocs.extend((
                AddressRelocation(0, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, alloc_symbols[0].id, 0),
                AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, alloc_symbols[1].id, 0),
            ))
        base = len(recv_records)
        recv_records.extend((recv_record, wait_record))
        recv_address_relocs.append(AddressRelocation(base, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, dsym.id, 0))
        recv_fragment = _fragment(intent, (recv_unit.id, wait_unit.id), recv_unit.logical_core, tuple(recv_records), (fsm, token, recv_peer), tuple(recv_symbols), (
            RuntimeRelocation(base, RuntimeOperandField.DTE_FSM, fsm.id), RuntimeRelocation(base, RuntimeOperandField.DTE_TOKEN, token.id), RuntimeRelocation(base, RuntimeOperandField.PEER_CORE, recv_peer.id), RuntimeRelocation(base + 1, RuntimeOperandField.DTE_TOKEN, token.id)), tuple(recv_address_relocs), (destination_abi,))
        return send_fragment, recv_fragment

    fragments.extend(transfer(us, ur, uw, abi[us.input_buffer_abi_refs[0]], abi[ur.output_buffer_abi_ref], "upload", allocate_destination=True))

    inputs = tuple(abi[item] for item in reduce.input_buffer_abi_refs)
    output = abi[reduce.output_buffer_abi_ref]
    source_symbol = _address(inputs[0], identity=tuple(item.binding_id for item in inputs))
    destination_symbol = _address(output)
    reduce_record = RelocatableRecord(reduce.id, RecordOpcode.LOCAL_REDUCE, (
        RecordOperand.literal("input_dtype", 1), RecordOperand.literal("accumulator_dtype", 1), RecordOperand.literal("output_dtype", 1),
        RecordOperand.literal("reduce_op", 1), RecordOperand.literal("rounding", 0), RecordOperand.literal("order", 0),
        RecordOperand.literal("input_count", 2), RecordOperand.literal("element_count", 512), RecordOperand.literal("input_stride_bytes", 2048),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination_symbol.id),
    ))
    fragments.append(_fragment(intent, (reduce.id,), reduce.logical_core, (reduce_record,), (), (source_symbol, destination_symbol), (), (
        AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
        AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0)), (*inputs, output)))
    fragments.extend(transfer(ds, dr, dw, abi[ds.input_buffer_abi_refs[0]], abi[dr.output_buffer_abi_ref], "download", free_after_send=(intent.scratch_buffer_abis[1], intent.scratch_buffer_abis[0])))
    return tuple(sorted(fragments, key=lambda item: item.id))


def lower_s2_lite_rooted_ar(intent: S2LiteRootedArN6Intent) -> S2LiteRootedArLoweredProgram:
    if type(intent) is not S2LiteRootedArN6Intent:
        raise SchemaError("must be an S2LiteRootedArN6Intent", path="intent")
    intent.validate("intent")
    dependencies = _resolve_dependencies(None, None, None, None, None)
    local = tuple(
        _leaf_fragment(item)
        for context in intent.lowering_contexts
        for item in _lower_fragments(context, dependencies)
    )
    return S2LiteRootedArLoweredProgram.create(intent=intent, local_fragments=local, overlay_fragments=_overlay(intent))


__all__ = ["lower_s2_lite_rooted_ar"]
