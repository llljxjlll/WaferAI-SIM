"""Exact public record lowerers for the isolated S3-Lite MoE intent."""

from __future__ import annotations

from collections import defaultdict
from math import prod

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    BufferABI,
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
    StateABI,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir0 import GemmWorkload, SwiGluWorkload
from ..schema.ir1 import SramAllocator
from ..schema.lite_moe_execution import (
    LiteMoeGlobalAction,
    LiteMoeGlobalDag,
    LiteMoeScheduled,
    LiteMoeTaskKind,
)
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.lite_moe_n6 import (
    LiteMoeBufferOperand,
    LiteMoeComputeUnit,
    LiteMoeDteUnit,
    LiteMoeN6Intent,
    LiteMoeStateLoadUnit,
)


_PRODUCER_PASS = "lite_moe_lowering"
_RUNTIME_FIELD_ORDER = {
    field: index for index, field in enumerate(RuntimeOperandField)
}


def _unique_by_id(values: tuple[object, ...]) -> tuple[object, ...]:
    result: dict[str, object] = {}
    for value in values:
        value_id = getattr(value, "id")
        previous = result.setdefault(value_id, value)
        if previous != value:
            raise SchemaError("content-addressed id collision", path="lite_moe")
    return tuple(result[key] for key in sorted(result))


def _indices(
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> tuple[
    dict[str, BufferABI],
    dict[str, LiteMoeGlobalAction],
    dict[str, object],
    dict[int, object],
]:
    intent.validate("intent")
    global_dag.validate("global_dag")
    schedule.validate("schedule")
    source.validate("source")
    if (
        intent.source_global_id != global_dag.id
        or intent.source_schedule_id != schedule.id
        or intent.source_n4_id != source.id
        or global_dag.source_schedule_id != schedule.id
        or schedule.source_n4_id != source.id
    ):
        raise SchemaError("Lite-MoE lowering sources disagree", path="source")
    actions = {action.id: action for action in global_dag.actions}
    placements = {placement.task_ref: placement for placement in schedule.placements}
    dies = {die.id: die for die in source.graph.fabric.dies}
    return (
        {abi.id: abi for abi in intent.buffer_abis},
        actions,
        placements,
        dies,
    )


def _action_core(
    action: LiteMoeGlobalAction,
    source: LiteMoeN4IR1,
) -> tuple[LogicalCoreRef, int]:
    die = next(
        (candidate for candidate in source.graph.fabric.dies if candidate.id == action.die_id),
        None,
    )
    if die is None:
        raise SchemaError("action references an unknown die", path="action.die_id")
    core = next((candidate for candidate in die.cores if candidate.id == action.core_ref), None)
    if core is None:
        raise SchemaError("action references an unknown core", path="action.core_ref")
    return LogicalCoreRef(action.die_id, core.local_core_id), core.runtime_core_id


def _runtime_symbol(
    kind: RuntimeSymbolKind,
    source_ref: str,
    identity: object,
) -> RuntimeSymbol:
    semantic = {"kind": kind.value, "source_ref": source_ref, "identity": identity}
    return RuntimeSymbol(
        stable_artifact_id(
            "runtime_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        kind,
        source_ref,
    )


def _absolute_symbol(abi: BufferABI) -> ProgramSymbol:
    semantic = {
        "schedule_id": abi.schedule_id,
        "binding_id": abi.binding_id,
        "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        abi.binding_id,
    )


def _label_symbol(abi: BufferABI, runtime_core_id: int) -> ProgramSymbol:
    semantic = {
        "schedule_id": abi.schedule_id,
        "runtime_core_id": runtime_core_id,
        "storage_id": abi.storage_id,
        "kind": int(ProgramSymbolKind.SRAM_LABEL),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.SRAM_LABEL,
        abi.storage_id,
    )


def _region_symbol(region_ref: str) -> ProgramSymbol:
    semantic = {
        "region_ref": region_ref,
        "kind": int(ProgramSymbolKind.SRAM_REGION),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.SRAM_REGION,
        region_ref,
    )


def _hbm_symbol(hbm_binding_ref: str) -> ProgramSymbol:
    semantic = {
        "hbm_binding_ref": hbm_binding_ref,
        "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        hbm_binding_ref,
    )


def _operand_abi(
    operand: LiteMoeBufferOperand,
    abi_index: dict[str, BufferABI],
    *,
    path: str,
) -> BufferABI:
    operand.validate(path)
    abi = abi_index.get(operand.buffer_abi_ref)
    if abi is None:
        raise SchemaError("operand references an unknown BufferABI", path=path)
    if operand.offset_bytes > abi.size_bytes or operand.size_bytes > abi.size_bytes - operand.offset_bytes:
        raise SchemaError("operand view exceeds BufferABI", path=path)
    return abi


def _region_for_abi(abi: BufferABI, source: LiteMoeN4IR1):
    die = next(item for item in source.graph.fabric.dies if item.id == abi.logical_core.die_id)
    core = next(item for item in die.cores if item.local_core_id == abi.logical_core.local_core_id)
    profile = next(
        item for item in source.graph.fabric.sram_profiles
        if item.id == core.sram_profile_ref
    )
    region = next((item for item in profile.regions if item.id == abi.region_ref), None)
    if (
        region is None
        or region.allocator is not SramAllocator.BLOCK
        or abi.region_offset_bytes + abi.size_bytes > region.size_bytes
    ):
        raise SchemaError("BufferABI does not fit one BLOCK SRAM region", path="buffer_abi")
    return region


def _decorate_lifecycle(
    fragment: CommandFragment,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> CommandFragment:
    actions = {action.id: action for action in global_dag.actions}
    placements = {placement.task_ref: placement for placement in schedule.placements}
    abi_by_binding = {abi.binding_id: abi for abi in fragment.buffer_abi}
    symbols = {symbol.id: symbol for symbol in fragment.program_symbols}
    streams: list[CoreFragmentStream] = []
    for stream in fragment.core_streams:
        runtime_by_record: dict[int, list[RuntimeRelocation]] = defaultdict(list)
        address_by_record: dict[int, list[AddressRelocation]] = defaultdict(list)
        for relocation in stream.runtime_relocations:
            runtime_by_record[relocation.record_index].append(relocation)
        for relocation in stream.address_relocations:
            address_by_record[relocation.record_index].append(relocation)
        records: list[RelocatableRecord] = []
        runtime_relocations: list[RuntimeRelocation] = []
        address_relocations: list[AddressRelocation] = []
        cursor = 0
        while cursor < len(stream.records):
            action_id = stream.records[cursor].source_global_action_id
            end = cursor + 1
            while end < len(stream.records) and stream.records[end].source_global_action_id == action_id:
                end += 1
            action = actions.get(action_id)
            if action is None:
                raise SchemaError("fragment references an unknown Lite-MoE action", path="fragment")
            logical_core, runtime_core_id = _action_core(action, source)
            placement = placements.get(action.task_ref)
            if (
                logical_core != stream.logical_core
                or placement is None
                or placement.die_id != action.die_id
                or placement.core_ref != action.core_ref
            ):
                raise SchemaError("fragment action/core placement mismatch", path="fragment")
            used = {
                abi_by_binding[use.binding_ref].storage_id: abi_by_binding[use.binding_ref]
                for use in action.buffer_uses
                if use.binding_ref in abi_by_binding
            }
            if len(used) != len({use.binding_ref for use in action.buffer_uses}):
                raise SchemaError("action buffer use lacks its leaf BufferABI", path="fragment.buffer_abi")
            starts = tuple(sorted(
                (abi for abi in used.values() if abi.lifetime_start == placement.ordinal),
                key=lambda abi: (abi.region_ref, abi.region_offset_bytes, abi.storage_id, abi.id),
            ))
            ends = tuple(reversed(sorted(
                (abi for abi in used.values() if abi.lifetime_end_exclusive == placement.ordinal + 1),
                key=lambda abi: (abi.region_ref, abi.region_offset_bytes, abi.storage_id, abi.id),
            )))
            for abi in starts:
                region = _region_for_abi(abi, source)
                region_symbol = _region_symbol(abi.region_ref)
                label_symbol = _label_symbol(abi, runtime_core_id)
                symbols[region_symbol.id] = region_symbol
                symbols[label_symbol.id] = label_symbol
                record_index = len(records)
                records.append(RelocatableRecord(
                    action.id,
                    RecordOpcode.SRAM_ALLOC_AT,
                    (
                        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol.id),
                        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label_symbol.id),
                        RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
                        RecordOperand.literal("size_bytes", abi.size_bytes),
                        RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
                        RecordOperand.literal("lifetime", 0),
                        RecordOperand.literal("spillable", region.spillable),
                    ),
                ))
                address_relocations.extend((
                    AddressRelocation(record_index, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, region_symbol.id, 0),
                    AddressRelocation(record_index, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, label_symbol.id, 0),
                ))
            for old_index in range(cursor, end):
                new_index = len(records)
                records.append(stream.records[old_index])
                runtime_relocations.extend(
                    RuntimeRelocation(new_index, relocation.field, relocation.symbol_ref)
                    for relocation in runtime_by_record[old_index]
                )
                address_relocations.extend(
                    AddressRelocation(new_index, relocation.operand_id, relocation.symbol_kind, relocation.symbol_ref, relocation.addend)
                    for relocation in address_by_record[old_index]
                )
            for abi in ends:
                label_symbol = _label_symbol(abi, runtime_core_id)
                symbols[label_symbol.id] = label_symbol
                record_index = len(records)
                records.append(RelocatableRecord(
                    action.id,
                    RecordOpcode.SRAM_FREE,
                    (RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label_symbol.id),),
                ))
                address_relocations.append(AddressRelocation(
                    record_index, SemanticOperandId.SYMBOL,
                    ProgramSymbolKind.SRAM_LABEL, label_symbol.id, 0,
                ))
            cursor = end
        streams.append(CoreFragmentStream(
            stream.logical_core,
            tuple(records),
            tuple(sorted(runtime_relocations, key=lambda item: (item.record_index, _RUNTIME_FIELD_ORDER[item.field]))),
            tuple(sorted(address_relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ))
    result = CommandFragment.create(
        producer_pass=fragment.producer_pass,
        source_global_dag_id=fragment.source_global_dag_id,
        kind=fragment.kind,
        claimed_action_ids=fragment.claimed_action_ids,
        core_streams=tuple(streams),
        runtime_symbols=fragment.runtime_symbols,
        program_symbols=tuple(sorted(symbols.values(), key=lambda item: item.id)),
        buffer_abi=fragment.buffer_abi,
        state_abi=fragment.state_abi,
    )
    result.validate("fragment")
    return result


def lower_lite_moe_compute_unit(
    unit: LiteMoeComputeUnit,
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> CommandFragment:
    """Lower one exact typed compute unit, preserving all BufferABI views."""

    if type(unit) is not LiteMoeComputeUnit:
        raise SchemaError("must be a LiteMoeComputeUnit", path="unit")
    abi_index, actions, _placements, _dies = _indices(
        intent, global_dag, schedule, source
    )
    if unit not in intent.compute_units:
        raise SchemaError("unit is not owned by the intent", path="unit")
    unit.validate("unit")
    action = actions.get(unit.action_ref)
    if action is None or action.kind is not unit.kind:
        raise SchemaError("compute unit/action mismatch", path="unit.action_ref")
    logical_core, runtime_core_id = _action_core(action, source)
    input_abis = tuple(
        _operand_abi(operand, abi_index, path=f"unit.inputs[{index}]")
        for index, operand in enumerate(unit.inputs)
    )
    output_abi = _operand_abi(unit.output, abi_index, path="unit.output")
    if any(abi.logical_core != logical_core for abi in (*input_abis, output_abi)):
        raise SchemaError("compute views must reside on the action core", path="unit")
    if any(abi.dtype is not DType.FP16 for abi in (*input_abis, output_abi)):
        raise SchemaError("Lite-MoE compute requires FP16 BufferABI", path="unit")

    if unit.kind is LiteMoeTaskKind.GEMM:
        if type(unit.workload) is not GemmWorkload:
            raise SchemaError("GEMM unit requires GemmWorkload", path="unit.workload")
        rank_m, rank_n, rank_k = unit.workload.rank_shape
        expected_sizes = (2 * rank_m * rank_k, 2 * rank_k * rank_n, 2 * rank_m * rank_n)
        if tuple(operand.size_bytes for operand in (*unit.inputs, unit.output)) != expected_sizes:
            raise SchemaError("GEMM view byte spans disagree with rank shape", path="unit")
        opcode = RecordOpcode.MATMUL
        parameters = (1, rank_m, rank_k, rank_n)
        data_symbol = _absolute_symbol(input_abis[1])
        data_operand = RecordOperand.address(
            "data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, data_symbol.id
        )
    else:
        if type(unit.workload) is not SwiGluWorkload:
            raise SchemaError("SWIGLU unit requires SwiGluWorkload", path="unit.workload")
        expected_sizes = (
            2 * prod(unit.workload.rank_input_shape),
            2 * prod(unit.workload.rank_output_shape),
        )
        if tuple(operand.size_bytes for operand in (*unit.inputs, unit.output)) != expected_sizes:
            raise SchemaError("SWIGLU view byte spans disagree with rank shape", path="unit")
        opcode = RecordOpcode.SWIGLU
        parameters = (prod(unit.workload.rank_output_shape),)
        data_symbol = None
        data_operand = RecordOperand.literal("data_address", 0)

    input_label = _label_symbol(input_abis[0], runtime_core_id)
    output_label = _label_symbol(output_abi, runtime_core_id)
    input_symbol = _absolute_symbol(input_abis[0])
    output_symbol = _absolute_symbol(output_abi)
    bind = RelocatableRecord(
        action.id,
        RecordOpcode.SRAM_BIND,
        (
            RecordOperand.literal("input_count", 1),
            RecordOperand.address(
                "input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0,
                input_label.id,
            ),
            *(
                RecordOperand.literal(f"input_label_{index}", 0)
                for index in range(1, 16)
            ),
            RecordOperand.address(
                "output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                output_label.id,
            ),
        ),
    )
    compute = RelocatableRecord(
        action.id,
        opcode,
        (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address(
                "input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_symbol.id,
            ),
            data_operand,
            RecordOperand.address(
                "output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_symbol.id,
            ),
            RecordOperand.literal("parameters", parameters),
        ),
    )
    relocations = [
        AddressRelocation(
            0, SemanticOperandId.SRAM_BIND_INPUT_0,
            ProgramSymbolKind.SRAM_LABEL, input_label.id, 0,
        ),
        AddressRelocation(
            0, SemanticOperandId.SRAM_BIND_OUTPUT,
            ProgramSymbolKind.SRAM_LABEL, output_label.id, 0,
        ),
        AddressRelocation(
            1, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS, input_symbol.id,
            unit.inputs[0].offset_bytes,
        ),
        AddressRelocation(
            1, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS, output_symbol.id,
            unit.output.offset_bytes,
        ),
    ]
    if data_symbol is not None:
        relocations.append(AddressRelocation(
            1, SemanticOperandId.COMPUTE_DATA_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS, data_symbol.id,
            unit.inputs[1].offset_bytes,
        ))
    symbols = (
        input_label,
        output_label,
        input_symbol,
        output_symbol,
        *((data_symbol,) if data_symbol is not None else ()),
    )
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER_PASS,
        source_global_dag_id=global_dag.id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            logical_core,
            (bind, compute),
            (),
            tuple(sorted(relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ),),
        runtime_symbols=(),
        program_symbols=_unique_by_id(symbols),
        buffer_abi=tuple(sorted(
            {abi.id: abi for abi in (*input_abis, output_abi)}.values(),
            key=lambda abi: abi.id,
        )),
        state_abi=(),
    )
    return _decorate_lifecycle(fragment, global_dag, schedule, source)


def lower_lite_moe_state_load_unit(
    unit: LiteMoeStateLoadUnit,
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> CommandFragment:
    """Lower one whole-state blocking HBM-to-SRAM load."""

    if type(unit) is not LiteMoeStateLoadUnit:
        raise SchemaError("must be a LiteMoeStateLoadUnit", path="unit")
    abi_index, actions, _placements, _dies = _indices(
        intent, global_dag, schedule, source
    )
    if unit not in intent.state_loads:
        raise SchemaError("unit is not owned by the intent", path="unit")
    unit.validate("unit")
    action = actions.get(unit.action_ref)
    if (
        action is None
        or action.kind is not LiteMoeTaskKind.DMA_IN
        or action.hbm_binding_ref != unit.hbm_binding_ref
    ):
        raise SchemaError("state-load unit/action mismatch", path="unit.action_ref")
    logical_core, _runtime_core_id = _action_core(action, source)
    local_abi = abi_index.get(unit.destination_buffer_abi_ref)
    if local_abi is None:
        raise SchemaError("state load references unknown BufferABI", path="unit.destination_buffer_abi_ref")
    if (
        local_abi.logical_core != logical_core
        or local_abi.dtype is not DType.FP16
        or local_abi.size_bytes != unit.bytes
    ):
        raise SchemaError("state-load destination must be one whole FP16 local buffer", path="unit")
    manifest = source.graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError("state load requires persistent-state manifest", path="source.graph")
    binding = next((item for item in manifest.bindings if item.id == unit.hbm_binding_ref), None)
    declaration = next((item for item in manifest.declarations if item.id == unit.state_ref), None)
    if (
        binding is None
        or declaration is None
        or binding.state_ref != declaration.id
        or binding.die_id != action.die_id
        or binding.size_bytes != unit.bytes
        or declaration.tensor_bytes != unit.bytes
    ):
        raise SchemaError("state-load HBM/declaration closure mismatch", path="unit")
    home = next((item for item in manifest.address_spaces if item.die_id == binding.die_id), None)
    if home is None:
        raise SchemaError("state load has no HBM address space", path="unit.hbm_binding_ref")

    hbm_symbol = _hbm_symbol(binding.id)
    local_symbol = _absolute_symbol(local_abi)
    record = RelocatableRecord(
        action.id,
        RecordOpcode.LSU_LOAD,
        (
            RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm_symbol.id),
            RecordOperand.literal("size_bytes", unit.bytes),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, local_symbol.id),
        ),
    )
    state_abi = StateABI.create(
        state_ref=declaration.id,
        hbm_binding_ref=binding.id,
        kind=declaration.identity.kind,
        lifetime=declaration.lifetime,
        access=declaration.access,
        shape=declaration.shape,
        dtype=declaration.dtype,
        layout=declaration.layout,
        die_id=binding.die_id,
        address=binding.address,
        size_bytes=binding.size_bytes,
        alignment_bytes=home.alignment_bytes,
    )
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER_PASS,
        source_global_dag_id=global_dag.id,
        kind=FragmentKind.STATE_IO,
        claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            logical_core,
            (record,),
            (),
            tuple(sorted((
                AddressRelocation(
                    0, SemanticOperandId.HBM_ADDRESS,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm_symbol.id, 0,
                ),
                AddressRelocation(
                    0, SemanticOperandId.DESTINATION_ADDRESS,
                    ProgramSymbolKind.ABSOLUTE_ADDRESS, local_symbol.id, 0,
                ),
            ), key=lambda item: int(item.operand_id))),
        ),),
        runtime_symbols=(),
        program_symbols=_unique_by_id((hbm_symbol, local_symbol)),
        buffer_abi=(local_abi,),
        state_abi=(state_abi,),
    )
    return _decorate_lifecycle(fragment, global_dag, schedule, source)


def _dte_record(
    *,
    action_id: str,
    is_send: bool,
    length_bytes: int,
    address: ProgramSymbol,
    fsm: RuntimeSymbol,
    peer: RuntimeSymbol,
    token: RuntimeSymbol | None,
) -> RelocatableRecord:
    token_operand = (
        RecordOperand.literal("token", 0)
        if token is None
        else RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id)
    )
    if is_send:
        return RelocatableRecord(
            action_id,
            RecordOpcode.DTE_SEND,
            (
                RecordOperand.literal("mode", 0),
                RecordOperand.literal("source_space", 0),
                RecordOperand.literal("completion", 1),
                RecordOperand.literal("datatype", 0),
                RecordOperand.literal("reduce_op", 0),
                RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm.id),
                token_operand,
                RecordOperand.literal("length_bytes", length_bytes),
                RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, address.id),
                RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer.id),
                RecordOperand.literal("expected_sources", 0),
                RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0),
                RecordOperand.literal("collective_id", 0),
                RecordOperand.literal("epoch", 0),
            ),
        )
    if token is None:
        raise SchemaError("DTE RECV requires an async token", path="unit.token_symbol")
    return RelocatableRecord(
        action_id,
        RecordOpcode.DTE_RECV,
        (
            RecordOperand.literal("mode", 0),
            RecordOperand.literal("completion", 0),
            RecordOperand.literal("datatype", 0),
            RecordOperand.literal("reduce_op", 0),
            RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm.id),
            token_operand,
            RecordOperand.literal("length_bytes", length_bytes),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, address.id),
            RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer.id),
            RecordOperand.literal("expected_sources", 0),
            RecordOperand.literal("tree_id", 0),
            RecordOperand.literal("group_id", 0),
            RecordOperand.literal("collective_id", 0),
            RecordOperand.literal("epoch", 0),
        ),
    )


def lower_lite_moe_dte_unit(
    unit: LiteMoeDteUnit,
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> tuple[CommandFragment, CommandFragment]:
    """Lower one route-bound flow into source and destination endpoint leaves."""

    if type(unit) is not LiteMoeDteUnit:
        raise SchemaError("must be a LiteMoeDteUnit", path="unit")
    abi_index, actions, _placements, _dies = _indices(
        intent, global_dag, schedule, source
    )
    if unit not in intent.dte_units:
        raise SchemaError("unit is not owned by the intent", path="unit")
    unit.validate("unit")
    send = actions.get(unit.send_action_ref)
    recv = actions.get(unit.recv_action_ref)
    wait = actions.get(unit.wait_action_ref)
    if (
        send is None or recv is None or wait is None
        or (send.kind, recv.kind, wait.kind)
        != (LiteMoeTaskKind.SEND, LiteMoeTaskKind.RECV, LiteMoeTaskKind.WAIT)
        or (send.flow_ref, recv.flow_ref, wait.flow_ref) != (unit.flow_ref,) * 3
        or (send.die_id, recv.die_id, wait.die_id)
        != (unit.source_die_id, unit.destination_die_id, unit.destination_die_id)
        or wait.deps != (recv.id,)
    ):
        raise SchemaError("DTE unit/action closure mismatch", path="unit")
    p2p_binding = next(
        (item for item in source.p2p_bindings if item.id == unit.p2p_binding_ref),
        None,
    )
    routes = tuple(
        route for group in source.graph.groups for route in group.embedding.routes
    )
    route = next((item for item in routes if item.id == unit.pair_route_ref), None)
    if (
        p2p_binding is None
        or route is None
        or (p2p_binding.source_die_id, p2p_binding.destination_die_id)
        != (unit.source_die_id, unit.destination_die_id)
        or (route.die_path[0], route.die_path[-1])
        != (unit.source_die_id, unit.destination_die_id)
    ):
        raise SchemaError("DTE unit route/binding closure mismatch", path="unit")
    source_abi = abi_index.get(unit.source_buffer_abi_ref)
    destination_abi = abi_index.get(unit.destination_buffer_abi_ref)
    source_core, _ = _action_core(send, source)
    destination_core, _ = _action_core(recv, source)
    if (
        source_abi is None or destination_abi is None
        or source_abi.logical_core != source_core
        or destination_abi.logical_core != destination_core
        or source_abi.dtype is not DType.FP16
        or destination_abi.dtype is not DType.FP16
        or source_abi.size_bytes != unit.bytes
        or destination_abi.size_bytes != unit.bytes
    ):
        raise SchemaError("DTE unit requires exact whole FP16 endpoint buffers", path="unit")

    fsm = _runtime_symbol(
        RuntimeSymbolKind.DTE_FSM,
        unit.channel_symbol,
        ("channel", unit.channel_symbol),
    )
    token = _runtime_symbol(
        RuntimeSymbolKind.DTE_TOKEN,
        unit.token_symbol,
        ("recv", recv.id),
    )
    source_peer = _runtime_symbol(
        RuntimeSymbolKind.RUNTIME_CORE,
        unit.channel_symbol,
        ("peer", send.id),
    )
    destination_peer = _runtime_symbol(
        RuntimeSymbolKind.RUNTIME_CORE,
        unit.channel_symbol,
        ("peer", recv.id),
    )
    source_address = _absolute_symbol(source_abi)
    destination_address = _absolute_symbol(destination_abi)
    send_record = _dte_record(
        action_id=send.id,
        is_send=True,
        length_bytes=unit.bytes,
        address=source_address,
        fsm=fsm,
        peer=source_peer,
        token=None,
    )
    recv_record = _dte_record(
        action_id=recv.id,
        is_send=False,
        length_bytes=unit.bytes,
        address=destination_address,
        fsm=fsm,
        peer=destination_peer,
        token=token,
    )
    wait_record = RelocatableRecord(
        wait.id,
        RecordOpcode.DTE_WAIT,
        (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
    )

    source_fragment = CommandFragment.create(
        producer_pass=_PRODUCER_PASS,
        source_global_dag_id=global_dag.id,
        kind=FragmentKind.MOE_TRANSFER,
        claimed_action_ids=(send.id,),
        core_streams=(CoreFragmentStream(
            source_core,
            (send_record,),
            tuple(sorted((
                RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, source_peer.id),
            ), key=lambda item: _RUNTIME_FIELD_ORDER[item.field])),
            (AddressRelocation(
                0, SemanticOperandId.SOURCE_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS, source_address.id, 0,
            ),),
        ),),
        runtime_symbols=_unique_by_id((fsm, source_peer)),
        program_symbols=(source_address,),
        buffer_abi=(source_abi,),
        state_abi=(),
    )
    destination_fragment = CommandFragment.create(
        producer_pass=_PRODUCER_PASS,
        source_global_dag_id=global_dag.id,
        kind=FragmentKind.MOE_TRANSFER,
        claimed_action_ids=tuple(sorted((recv.id, wait.id))),
        core_streams=(CoreFragmentStream(
            destination_core,
            (recv_record, wait_record),
            (
                RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
                RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, destination_peer.id),
                RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
            ),
            (AddressRelocation(
                0, SemanticOperandId.DESTINATION_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_address.id, 0,
            ),),
        ),),
        runtime_symbols=_unique_by_id((fsm, token, destination_peer)),
        program_symbols=(destination_address,),
        buffer_abi=(destination_abi,),
        state_abi=(),
    )
    return (
        _decorate_lifecycle(source_fragment, global_dag, schedule, source),
        _decorate_lifecycle(destination_fragment, global_dag, schedule, source),
    )


def lower_lite_moe_n6_intent(
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
) -> tuple[CommandFragment, ...]:
    """Lower all 80 intent actions into canonical leaf fragments."""

    _indices(intent, global_dag, schedule, source)
    fragments = [
        lower_lite_moe_state_load_unit(unit, intent, global_dag, schedule, source)
        for unit in intent.state_loads
    ]
    fragments.extend(
        lower_lite_moe_compute_unit(unit, intent, global_dag, schedule, source)
        for unit in intent.compute_units
    )
    for unit in intent.dte_units:
        fragments.extend(
            lower_lite_moe_dte_unit(unit, intent, global_dag, schedule, source)
        )
    result = tuple(sorted(fragments, key=lambda fragment: fragment.id))
    claimed = tuple(
        action_id for fragment in result for action_id in fragment.claimed_action_ids
    )
    if len(result) != 72 or set(claimed) != {action.id for action in global_dag.actions} or len(claimed) != 80:
        raise SchemaError("Lite-MoE fragments must cover 80 actions once in 72 leaves", path="fragments")
    return result


def validate_lite_moe_fragment(
    fragment: CommandFragment,
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    source: LiteMoeN4IR1,
    path: str = "command_fragment",
) -> None:
    """Require byte-exact equality with one canonical Lite-MoE intent leaf."""

    if type(fragment) is not CommandFragment:
        raise SchemaError("must be a CommandFragment", path=path)
    fragment.validate(path)
    candidates = tuple(
        candidate
        for candidate in lower_lite_moe_n6_intent(
            intent, global_dag, schedule, source
        )
        if candidate.claimed_action_ids == fragment.claimed_action_ids
    )
    if len(candidates) != 1 or fragment != candidates[0]:
        raise SchemaError("fragment is not the exact Lite-MoE intent lowering", path=path)


__all__ = [
    "lower_lite_moe_compute_unit",
    "lower_lite_moe_dte_unit",
    "lower_lite_moe_n6_intent",
    "lower_lite_moe_state_load_unit",
    "validate_lite_moe_fragment",
]
