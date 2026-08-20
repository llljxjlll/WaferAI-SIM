"""Production leaf lowering for the fixed S2-Lite DP4 tree AllReduce."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.lifecycle import _core_runtime_id, _storage_label_symbol
from ..schema.artifact_manifest import (
    AddressRelocation,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbolKind,
    SemanticOperandId,
    ProgramSymbolKind,
)
from ..schema.lite_train_dp4 import TreeArFlowKind, TreeArReduceKind
from ..schema.lite_train_dp4_n6 import (
    Dp4TreeExecutableKind,
    S2LiteDp4TreeArLoweredProgram,
    S2LiteDp4TreeArN6Intent,
)
from ..schema.n6 import _leaf_fragment
from .lite_train_rooted_ar_lower_program import (
    _address,
    _alloc_record,
    _dte_record,
    _fragment,
    _runtime,
)
from .lower_program import _lower_fragments, _resolve_dependencies


def _free(action_id, abi, context):
    symbol = _storage_label_symbol(
        abi.schedule_id,
        _core_runtime_id(context, abi.logical_core),
        abi.storage_id,
    )
    record = RelocatableRecord(
        action_id,
        RecordOpcode.SRAM_FREE,
        (RecordOperand.address("symbol", SemanticOperandId.SYMBOL, symbol.id),),
    )
    relocation = AddressRelocation(
        0,
        SemanticOperandId.SYMBOL,
        ProgramSymbolKind.SRAM_LABEL,
        symbol.id,
        0,
    )
    return record, symbol, relocation


def _overlay(intent: S2LiteDp4TreeArN6Intent):
    abis = {
        item.id: item
        for item in intent.gradient_buffer_abis + intent.scratch_buffer_abis
    }
    contexts = {index: context for index, context in enumerate(intent.lowering_contexts)}
    units = intent.units
    by_source: dict[str, list] = {}
    for unit in units:
        by_source.setdefault(unit.source_ref, []).append(unit)
    fragments = []

    # Two local copies allocate slice0 on die0/die2.
    for copy in (unit for unit in units if unit.kind is Dp4TreeExecutableKind.LOCAL_COPY):
        source = abis[copy.input_buffer_abi_refs[0]]
        destination = abis[copy.output_buffer_abi_ref]
        source_symbol, destination_symbol = _address(source), _address(destination)
        token = _runtime(RuntimeSymbolKind.DTE_TOKEN, copy.id, ("dp4_tree_copy", copy.id))
        alloc, alloc_symbols = _alloc_record(
            copy.id, destination, contexts[copy.logical_core.die_id]
        )
        issue = RelocatableRecord(
            copy.id,
            RecordOpcode.DTE_ISSUE,
            (
                RecordOperand.literal("direction", 0),
                RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
                RecordOperand.literal("payload_bits", copy.bytes * 8),
                RecordOperand.literal("size_bytes", copy.bytes),
                RecordOperand.literal("hbm_address", 0),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol.id),
                RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination_symbol.id),
            ),
        )
        wait = RelocatableRecord(
            copy.id,
            RecordOpcode.DTE_WAIT,
            (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
        )
        fragments.append(
            _fragment(
                intent,
                (copy.id,),
                copy.logical_core,
                (alloc, issue, wait),
                (token,),
                (*alloc_symbols, source_symbol, destination_symbol),
                (
                    RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
                    RuntimeRelocation(2, RuntimeOperandField.DTE_TOKEN, token.id),
                ),
                (
                    AddressRelocation(0, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, alloc_symbols[0].id, 0),
                    AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, alloc_symbols[1].id, 0),
                    AddressRelocation(1, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
                    AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0),
                ),
                (source, destination),
            )
        )

    flow_by_id = {flow.id: flow for flow in intent.source.tree_flows}
    scratch = intent.scratch_buffer_abis
    free_after_send = {
        TreeArFlowKind.BROADCAST_0_TO_2: scratch[0],
        TreeArFlowKind.BROADCAST_2_TO_3: scratch[2],
    }
    allocate_destination = {
        TreeArFlowKind.UPLOAD_1_TO_0,
        TreeArFlowKind.UPLOAD_3_TO_2,
    }

    for flow in intent.source.tree_flows:
        trio = by_source[flow.id]
        send = next(item for item in trio if item.kind is Dp4TreeExecutableKind.FLOW_SEND)
        recv = next(item for item in trio if item.kind is Dp4TreeExecutableKind.FLOW_RECV)
        wait = next(item for item in trio if item.kind is Dp4TreeExecutableKind.FLOW_WAIT)
        source = abis[send.input_buffer_abi_refs[0]]
        destination = abis[recv.output_buffer_abi_ref]
        fsm = _runtime(RuntimeSymbolKind.DTE_FSM, flow.channel_ref, ("dp4_tree", flow.kind.value, "fsm"))
        token = _runtime(RuntimeSymbolKind.DTE_TOKEN, flow.recv_token_ref, ("dp4_tree", flow.kind.value, "recv"))
        send_peer = _runtime(RuntimeSymbolKind.RUNTIME_CORE, recv.id, ("dp4_tree", flow.kind.value, "send_peer"))
        recv_peer = _runtime(RuntimeSymbolKind.RUNTIME_CORE, send.id, ("dp4_tree", flow.kind.value, "recv_peer"))
        source_symbol, destination_symbol = _address(source), _address(destination)
        send_records = [_dte_record(send, source_symbol, fsm, send_peer, None, send=True)]
        send_program_symbols = [source_symbol]
        send_abis = [source]
        send_address = [AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0)]
        if flow.kind in free_after_send:
            free_abi = free_after_send[flow.kind]
            free_record, free_symbol, free_relocation = _free(
                send.id, free_abi, contexts[send.logical_core.die_id]
            )
            send_records.append(free_record)
            send_program_symbols.append(free_symbol)
            send_abis.append(free_abi)
            send_address.append(
                AddressRelocation(
                    1,
                    free_relocation.operand_id,
                    free_relocation.symbol_kind,
                    free_relocation.symbol_ref,
                    0,
                )
            )
        fragments.append(
            _fragment(
                intent,
                (send.id,),
                send.logical_core,
                tuple(send_records),
                (fsm, send_peer),
                tuple(send_program_symbols),
                (
                    RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                    RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, send_peer.id),
                ),
                tuple(send_address),
                tuple(send_abis),
            )
        )
        recv_records = []
        recv_symbols = [destination_symbol]
        recv_address = []
        if flow.kind in allocate_destination:
            alloc, alloc_symbols = _alloc_record(
                recv.id, destination, contexts[recv.logical_core.die_id]
            )
            recv_records.append(alloc)
            recv_symbols.extend(alloc_symbols)
            recv_address.extend(
                (
                    AddressRelocation(0, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, alloc_symbols[0].id, 0),
                    AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, alloc_symbols[1].id, 0),
                )
            )
        base = len(recv_records)
        recv_records.extend(
            (
                _dte_record(recv, destination_symbol, fsm, recv_peer, token, send=False),
                RelocatableRecord(wait.id, RecordOpcode.DTE_WAIT, (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),)),
            )
        )
        recv_address.append(AddressRelocation(base, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0))
        fragments.append(
            _fragment(
                intent,
                (recv.id, wait.id),
                recv.logical_core,
                tuple(recv_records),
                (fsm, token, recv_peer),
                tuple(recv_symbols),
                (
                    RuntimeRelocation(base, RuntimeOperandField.DTE_FSM, fsm.id),
                    RuntimeRelocation(base, RuntimeOperandField.DTE_TOKEN, token.id),
                    RuntimeRelocation(base, RuntimeOperandField.PEER_CORE, recv_peer.id),
                    RuntimeRelocation(base + 1, RuntimeOperandField.DTE_TOKEN, token.id),
                ),
                tuple(recv_address),
                (destination,),
            )
        )

    reduce_by_id = {reduce.id: reduce for reduce in intent.source.tree_reduces}
    free_after_reduce = {
        TreeArReduceKind.PAIR_23: scratch[3],
        TreeArReduceKind.GLOBAL_AT_0: scratch[1],
    }
    for unit in (item for item in units if item.kind is Dp4TreeExecutableKind.LOCAL_REDUCE):
        reduce = reduce_by_id[unit.source_ref]
        inputs = tuple(abis[item] for item in unit.input_buffer_abi_refs)
        output = abis[unit.output_buffer_abi_ref]
        source_symbol = _address(inputs[0], identity=(unit.id, tuple(item.binding_id for item in inputs)))
        destination_symbol = _address(output)
        records = [RelocatableRecord(unit.id, RecordOpcode.LOCAL_REDUCE, (
            RecordOperand.literal("input_dtype", 1),
            RecordOperand.literal("accumulator_dtype", 1),
            RecordOperand.literal("output_dtype", 1),
            RecordOperand.literal("reduce_op", 1),
            RecordOperand.literal("rounding", 0),
            RecordOperand.literal("order", 0),
            RecordOperand.literal("input_count", 2),
            RecordOperand.literal("element_count", 512),
            RecordOperand.literal("input_stride_bytes", 2048),
            RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol.id),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination_symbol.id),
        ))]
        symbols = [source_symbol, destination_symbol]
        reduce_abis = [*inputs, output]
        address_relocations = [
            AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
            AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0),
        ]
        if reduce.kind in free_after_reduce:
            free_abi = free_after_reduce[reduce.kind]
            free_record, free_symbol, free_relocation = _free(
                unit.id, free_abi, contexts[unit.logical_core.die_id]
            )
            records.append(free_record)
            symbols.append(free_symbol)
            reduce_abis.append(free_abi)
            address_relocations.append(AddressRelocation(1, free_relocation.operand_id, free_relocation.symbol_kind, free_relocation.symbol_ref, 0))
        fragments.append(
            _fragment(
                intent,
                (unit.id,),
                unit.logical_core,
                tuple(records),
                (),
                tuple(symbols),
                (),
                tuple(address_relocations),
                tuple(reduce_abis),
            )
        )
    return tuple(sorted(fragments, key=lambda item: item.id))


def lower_s2_lite_dp4_tree_ar(
    intent: S2LiteDp4TreeArN6Intent,
) -> S2LiteDp4TreeArLoweredProgram:
    if type(intent) is not S2LiteDp4TreeArN6Intent:
        raise SchemaError("must be an S2LiteDp4TreeArN6Intent", path="intent")
    intent.validate("intent")
    dependencies = _resolve_dependencies(None, None, None, None, None)
    local = tuple(
        _leaf_fragment(fragment)
        for context in intent.lowering_contexts
        for fragment in _lower_fragments(context, dependencies)
    )
    return S2LiteDp4TreeArLoweredProgram.create(
        intent=intent,
        local_fragments=local,
        overlay_fragments=_overlay(intent),
    )


__all__ = ["lower_s2_lite_dp4_tree_ar"]
