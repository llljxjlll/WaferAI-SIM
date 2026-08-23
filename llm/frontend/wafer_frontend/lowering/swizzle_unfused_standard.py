"""Lower the explicit UNFUSED comparison DAG to standard command records."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    EventCredit,
    FragmentKind,
    FragmentInterface,
    LinkedCoreStream,
    LinkedProgramManifest,
    LinkedRecordRef,
    LogicalStartEvent,
    ManifestInputDigest,
    ManifestInputKind,
    ProgramControlEnvelope,
    ProgramFailurePolicy,
    ProgramSymbol,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_plan import SwizzleValueUse
from ..schema.serde import canonical_digest
from ..schema.swizzle_unfused import UnfusedComparisonPlan, UnfusedComparisonProjection
from ..schema.swizzle_unfused_abi import UnfusedComparisonCoreABI
from ..schema.swizzle_unfused_lowering import (
    UnfusedComparisonLoweredProgram,
    UnfusedComparisonOperandABI,
)
from ..schema.swizzle_unfused_standard import UnfusedComparisonStandardLinkedProgram
from .swizzle_standard import _dtype_code, _relocations, _region


_SCHEMA = "wafer_frontend.unfused_comparison_standard_lowering/v1alpha1"
_PRODUCER = "unfused_comparison_standard_lowering"


def _id(kind: str, semantic: object) -> str:
    return stable_artifact_id(
        f"unfused_comparison_standard_{kind}", semantic, schema_version=_SCHEMA
    )


def _derive_buffers(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
    core_abi: UnfusedComparisonCoreABI,
) -> tuple[BufferABI, ...]:
    actions = {action.id: action for program in plan.rank_programs for action in program.actions}
    orders = {item.task_ref: item.core_order for item in core_abi.task_bindings}
    storage = {(item.rank, item.storage_ref): item for item in core_abi.storage_bindings}
    grouped = defaultdict(list)
    for operand in projection.operands:
        grouped[(actions[operand.task_ref].rank, operand.storage_ref)].append(operand)
    terminal_tasks = {item.terminal_task_ref for item in projection.ranks}
    result = []
    for key, views in grouped.items():
        binding = storage[key]
        first = min(views, key=lambda item: (orders[item.task_ref], item.ordinal))
        computed_lifetime = (
            min(orders[item.task_ref] for item in views),
            max(orders[item.task_ref] for item in views) + 1,
        )
        lifetime = (
            binding.lifetime_start,
            binding.lifetime_end_exclusive,
        )
        if lifetime != computed_lifetime:
            raise SchemaError(
                "storage binding lifetime disagrees with exact typed uses",
                path="core_abi.storage_bindings",
            )
        storage_id = _id("storage", {"core_abi": core_abi.id, "key": key})
        root_binding_id = _id("storage_root_binding", {"core_abi": core_abi.id, "key": key})
        core = next(
            core
            for die in ir1.fabric.dies
            if die.id == binding.logical_core.die_id
            for core in die.cores
            if core.local_core_id == binding.logical_core.local_core_id
        )
        profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
        region = next(item for item in profile.regions if item.id == binding.region_ref)
        dtype_bytes = 2 if first.dtype is DType.FP16 else 4
        terminal_views = tuple(
            item
            for item in views
            if item.task_ref in terminal_tasks
            and item.use is SwizzleValueUse.WRITE
        )
        terminal = terminal_views[0] if len(terminal_views) == 1 else None
        if terminal is not None and binding.size_bytes != terminal.byte_extent:
            terminal = None
        root_semantic = {
            "schedule_id": core_abi.id,
            "binding_id": root_binding_id,
            "value_id": (
                terminal.source_tensor_ref
                if terminal is not None
                else binding.storage_ref
            ),
            "logical_core": binding.logical_core,
            "tensor_slice": (
                TensorSlice(
                    terminal.source_tensor_ref,
                    terminal.tensor_offset,
                    terminal.shape,
                )
                if terminal is not None
                else TensorSlice(
                    binding.storage_ref,
                    (0,),
                    (binding.size_bytes // dtype_bytes,),
                )
            ),
            "region_ref": binding.region_ref,
            "region_offset_bytes": binding.base_address - region.base_bytes,
            "size_bytes": binding.size_bytes,
            "alignment_bytes": binding.alignment_bytes,
            "banks": (),
            "storage_id": storage_id,
            "alias_of": None,
            "lifetime_start": lifetime[0],
            "lifetime_end_exclusive": lifetime[1],
            "dtype": first.dtype,
            "layout": "unfused_comparison_storage/v1",
            "ownership": (
                BufferOwnership.BORROWED
                if first.use is SwizzleValueUse.READ
                else BufferOwnership.OWNED
            ),
        }
        result.append(BufferABI(id=_id("buffer_abi", root_semantic), **root_semantic))
        by_value = defaultdict(list)
        for view in views:
            by_value[view.value_ref].append(view)
        for value_ref, value_views in sorted(by_value.items()):
            witness = value_views[0]
            if any(
                (item.shape, item.layout, item.dtype, item.byte_extent, item.byte_offset)
                != (witness.shape, witness.layout, witness.dtype, witness.byte_extent, witness.byte_offset)
                for item in value_views[1:]
            ):
                raise SchemaError("one value has inconsistent typed views", path="projection.operands")
            semantic = {
                "schedule_id": core_abi.id,
                "binding_id": _id("value_binding", {"core_abi": core_abi.id, "value": value_ref}),
                "value_id": value_ref,
                "logical_core": binding.logical_core,
                "tensor_slice": TensorSlice(value_ref, (0,) * len(witness.shape), witness.shape),
                "region_ref": binding.region_ref,
                "region_offset_bytes": binding.base_address - region.base_bytes + witness.byte_offset,
                "size_bytes": witness.byte_extent,
                "alignment_bytes": binding.alignment_bytes,
                "banks": (),
                "storage_id": storage_id,
                "alias_of": root_binding_id,
                "lifetime_start": min(orders[item.task_ref] for item in value_views),
                "lifetime_end_exclusive": max(orders[item.task_ref] for item in value_views) + 1,
                "dtype": witness.dtype,
                "layout": witness.layout,
                "ownership": BufferOwnership.ALIASED,
            }
            result.append(BufferABI(id=_id("buffer_abi", semantic), **semantic))
    return tuple(sorted(result, key=lambda item: item.id))


def _root_storage_refs(
    core_abi: UnfusedComparisonCoreABI,
) -> dict[str, str]:
    return {
        _id(
            "storage",
            {
                "core_abi": core_abi.id,
                "key": (item.rank, item.storage_ref),
            },
        ): item.storage_ref
        for item in core_abi.storage_bindings
    }


def lower_unfused_comparison_fragment(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
    lowered: UnfusedComparisonLoweredProgram,
    core_abi: UnfusedComparisonCoreABI,
    operand_abi: UnfusedComparisonOperandABI,
) -> CommandFragment:
    projection.validate_against(ir1, plan)
    lowered.validate_against(plan, projection)
    core_abi.validate_against(ir1, plan, projection)
    operand_abi.validate_against(ir1, plan, projection)
    actions = {action.id: action for program in plan.rank_programs for action in program.actions}
    task_bindings = {item.task_ref: item for item in core_abi.task_bindings}
    runtime_bindings = {item.task_ref: item for item in core_abi.runtime_bindings}
    storage_bindings = {(item.rank, item.storage_ref): item for item in core_abi.storage_bindings}
    views_by_task = defaultdict(list)
    for view in operand_abi.operands:
        views_by_task[view.task_ref].append(view)
    matmuls = {item.task_ref: item for item in operand_abi.matmul_contracts}
    dtes = {item.task_ref: item for item in operand_abi.dte_contracts}
    reduces = {item.task_ref: item for item in operand_abi.reduce_contracts}
    events_by_owner = defaultdict(list)
    for event in core_abi.barrier_events:
        events_by_owner[event.owner_task_ref].append(event)
    buffers = _derive_buffers(ir1, plan, projection, core_abi)
    root_by_storage = {item.storage_id: item for item in buffers if item.alias_of is None}
    root_storage_ref = _root_storage_refs(core_abi)
    alias_by_value = {item.value_id: item for item in buffers if item.alias_of is not None}
    program_symbols = {}
    runtime_symbols = {}

    def program(kind: ProgramSymbolKind, source_ref: str) -> str:
        symbol_id = _id("program_symbol", {"kind": kind, "source_ref": source_ref})
        program_symbols.setdefault(symbol_id, ProgramSymbol(symbol_id, kind, source_ref))
        return symbol_id

    def runtime(kind: RuntimeSymbolKind, symbol_id: str, source_ref: str) -> str:
        runtime_symbols.setdefault(symbol_id, RuntimeSymbol(symbol_id, kind, source_ref))
        return symbol_id

    abs_symbol = {
        value_ref: program(ProgramSymbolKind.ABSOLUTE_ADDRESS, abi.binding_id)
        for value_ref, abi in alias_by_value.items()
    }
    label_symbol = {
        storage_id: program(ProgramSymbolKind.SRAM_LABEL, storage_id)
        for storage_id in root_by_storage
    }
    region_symbol = {
        region_ref: program(ProgramSymbolKind.SRAM_REGION, region_ref)
        for region_ref in {item.region_ref for item in root_by_storage.values()}
    }
    records_by_core = defaultdict(list)
    roots_by_task = defaultdict(dict)
    for view in operand_abi.operands:
        abi = alias_by_value[view.value_ref]
        roots_by_task[view.task_ref][abi.storage_id] = root_by_storage[abi.storage_id]

    for program_rank in plan.rank_programs:
        for action in program_rank.actions:
            owner = action.id
            core = task_bindings[owner].logical_core
            records = records_by_core[core]
            views = sorted(views_by_task[owner], key=lambda item: item.ordinal)
            used_roots = roots_by_task[owner]
            for root in sorted(used_roots.values(), key=lambda item: item.id):
                if root.lifetime_start == task_bindings[owner].core_order:
                    storage_binding = storage_bindings[
                        (action.rank, root_storage_ref[root.storage_id])
                    ]
                    region = _region(ir1, storage_binding)
                    records.append(RelocatableRecord(owner, RecordOpcode.SRAM_ALLOC_AT, (
                        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol[root.region_ref]),
                        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label_symbol[root.storage_id]),
                        RecordOperand.literal("region_offset_bytes", root.region_offset_bytes),
                        RecordOperand.literal("size_bytes", root.size_bytes),
                        RecordOperand.literal("alignment_bytes", root.alignment_bytes),
                        RecordOperand.literal("lifetime", 0),
                        RecordOperand.literal("spillable", region.spillable),
                    )))
            if action.kind is SwizzleActionKind.COMP:
                contract = matmuls[owner]
                input0, weight, output = views
                bind_operands = [RecordOperand.literal("input_count", 1)]
                bind_operands.append(RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0, label_symbol[alias_by_value[input0.value_ref].storage_id]))
                bind_operands.extend(RecordOperand.literal(f"input_label_{index}", 0) for index in range(1, 16))
                bind_operands.append(RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, label_symbol[alias_by_value[output.value_ref].storage_id]))
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_BIND, tuple(bind_operands)))
                records.append(RelocatableRecord(owner, RecordOpcode.MATMUL, (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, abs_symbol[input0.value_ref]),
                    RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, abs_symbol[weight.value_ref]),
                    RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, abs_symbol[output.value_ref]),
                    RecordOperand.literal("parameters", (1, contract.m, contract.k, contract.n)),
                )))
            elif action.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                contract = dtes[owner]
                binding = runtime_bindings[owner]
                runtime(RuntimeSymbolKind.DTE_FSM, binding.fsm_symbol_ref, binding.flow_ref)
                runtime(RuntimeSymbolKind.RUNTIME_CORE, binding.peer_symbol_ref, str(binding.peer_core))
                tail = (
                    RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, binding.peer_symbol_ref),
                    RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                    RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0), RecordOperand.literal("epoch", 0),
                )
                view = views[0]
                if action.kind is SwizzleActionKind.SEND:
                    opcode = RecordOpcode.DTE_SEND
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0), RecordOperand.literal("completion", 1),
                        RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, binding.fsm_symbol_ref), RecordOperand.literal("token", 0),
                        RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, abs_symbol[view.value_ref]), *tail,
                    )
                else:
                    opcode = RecordOpcode.DTE_RECV
                    runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, owner)
                    operands = (
                        RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0), RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, binding.fsm_symbol_ref),
                        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref), RecordOperand.literal("length_bytes", contract.logical_bytes),
                        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[view.value_ref]), *tail,
                    )
                records.append(RelocatableRecord(owner, opcode, operands))
            elif action.kind is SwizzleActionKind.WAIT:
                binding = runtime_bindings[owner]
                runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, action.deps[0])
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                )))
            elif action.kind is SwizzleActionKind.LOCAL_COPY:
                contract = dtes[owner]
                binding = runtime_bindings[owner]
                runtime(RuntimeSymbolKind.DTE_TOKEN, binding.token_symbol_ref, owner)
                source, destination = views
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_ISSUE, (
                    RecordOperand.literal("direction", 0), RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                    RecordOperand.literal("payload_bits", contract.logical_bytes * 8), RecordOperand.literal("size_bytes", contract.logical_bytes), RecordOperand.literal("hbm_address", 0),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, abs_symbol[source.value_ref]),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[destination.value_ref]),
                )))
                records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, binding.token_symbol_ref),
                )))
            elif action.kind is SwizzleActionKind.REDUCE:
                contract = reduces[owner]
                source, accumulator, _output = views
                span_ref = _id("reduce_span", {"task": owner, "storage": source.storage_ref, "offset": source.byte_offset})
                source_symbol = program(ProgramSymbolKind.ABSOLUTE_ADDRESS, span_ref)
                records.append(RelocatableRecord(owner, RecordOpcode.LOCAL_REDUCE, (
                    RecordOperand.literal("input_dtype", _dtype_code(contract.dtype)),
                    RecordOperand.literal("accumulator_dtype", _dtype_code(contract.accumulation_dtype)),
                    RecordOperand.literal("output_dtype", _dtype_code(contract.dtype)),
                    RecordOperand.literal("reduce_op", 1), RecordOperand.literal("rounding", 0), RecordOperand.literal("order", 0),
                    RecordOperand.literal("input_count", contract.input_count), RecordOperand.literal("element_count", contract.element_count),
                    RecordOperand.literal("input_stride_bytes", contract.input_stride_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source_symbol),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_symbol[accumulator.value_ref]),
                )))
            elif action.kind is SwizzleActionKind.BARRIER:
                for event in sorted(events_by_owner[owner], key=lambda item: (item.opcode.value, item.event_symbol_ref)):
                    runtime(RuntimeSymbolKind.RUNTIME_CORE, event.source_core_symbol_ref, str(event.source_core))
                    runtime(RuntimeSymbolKind.RUNTIME_CORE, event.destination_core_symbol_ref, str(event.destination_core))
                    runtime(RuntimeSymbolKind.EVENT_TAG, event.event_symbol_ref, event.barrier_ref)
                    operands = (
                        RecordOperand.runtime("source_core", RuntimeOperandField.SOURCE_CORE, event.source_core_symbol_ref),
                        RecordOperand.runtime("destination_core", RuntimeOperandField.DESTINATION_CORE, event.destination_core_symbol_ref),
                        RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, event.event_symbol_ref),
                    )
                    if event.opcode is RecordOpcode.EVENT_WAIT:
                        operands = (*operands, RecordOperand.literal("count", 1))
                    records.append(RelocatableRecord(owner, event.opcode, operands))
            for root in reversed(sorted(used_roots.values(), key=lambda item: item.id)):
                if root.lifetime_end_exclusive == task_bindings[owner].core_order + 1:
                    records.append(RelocatableRecord(owner, RecordOpcode.SRAM_FREE, (
                        RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label_symbol[root.storage_id]),
                    )))
    streams = []
    for core, records in sorted(records_by_core.items(), key=lambda item: (item[0].die_id, item[0].local_core_id)):
        runtime_relocations, address_relocations = _relocations(records)
        streams.append(CoreFragmentStream(core, tuple(records), runtime_relocations, address_relocations))
    used_program_symbol_refs = {
        relocation.symbol_ref
        for stream in streams
        for relocation in stream.address_relocations
    }
    result = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=projection.id,
        kind=FragmentKind.UNFUSED_COMPARISON,
        claimed_action_ids=tuple(sorted(actions)),
        core_streams=tuple(streams),
        runtime_symbols=tuple(runtime_symbols[key] for key in sorted(runtime_symbols)),
        program_symbols=tuple(
            program_symbols[key]
            for key in sorted(used_program_symbol_refs)
        ),
        buffer_abi=buffers,
        state_abi=(),
    )
    result.validate()
    return result


def link_unfused_comparison_manifest(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
    lowered: UnfusedComparisonLoweredProgram,
    core_abi: UnfusedComparisonCoreABI,
    operand_abi: UnfusedComparisonOperandABI,
    fragment: CommandFragment,
) -> LinkedProgramManifest:
    expected = lower_unfused_comparison_fragment(
        ir1, plan, projection, lowered, core_abi, operand_abi
    )
    if fragment != expected:
        raise SchemaError("fragment is not the exact UNFUSED producer result", path="fragment")
    actions = {action.id: action for program in plan.rank_programs for action in program.actions}
    task_core = {item.task_ref: item.logical_core for item in core_abi.task_bindings}
    task_binding_by_core = {item.logical_core: item for item in core_abi.task_bindings}
    storage_binding = {(item.rank, item.storage_ref): item for item in core_abi.storage_bindings}
    root_storage_ref = _root_storage_refs(core_abi)
    buffer_by_id = {item.id: item for item in fragment.buffer_abi}
    buffer_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
    buffer_by_storage = defaultdict(list)
    for item in fragment.buffer_abi:
        buffer_by_storage[item.storage_id].append(item)
    symbol_by_id = {item.id: item for item in fragment.program_symbols}
    symbol_uses = defaultdict(set)
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            symbol_uses[relocation.symbol_ref].add(stream.logical_core)
    address_bindings = []
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            record = stream.records[relocation.record_index]
            symbol = symbol_by_id[relocation.symbol_ref]
            if symbol.kind is ProgramSymbolKind.SRAM_REGION:
                label_operand = next(
                    item for item in record.operands
                    if item.operand_id is SemanticOperandId.LABEL_SYMBOL
                )
                label = symbol_by_id[label_operand.symbol_ref]
                candidates = [
                    item for item in buffer_by_storage[label.source_ref]
                    if item.alias_of is None
                ]
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                candidates = [
                    item for item in buffer_by_storage[symbol.source_ref]
                    if item.alias_of is None
                ]
            elif symbol.source_ref in buffer_by_binding:
                candidates = [buffer_by_binding[symbol.source_ref]]
            elif (
                record.opcode is RecordOpcode.LOCAL_REDUCE
                and relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS
            ):
                views = sorted(
                    (item for item in operand_abi.operands if item.task_ref == record.source_global_action_id),
                    key=lambda item: item.ordinal,
                )
                candidates = [
                    next(item for item in fragment.buffer_abi if item.value_id == view.value_ref)
                    for view in views[:2]
                ]
            else:
                raise SchemaError(
                    "program symbol has no exact UNFUSED storage witness",
                    path="fragment.program_symbols",
                )
            key = (fragment.id, stream.logical_core, relocation.record_index, relocation.operand_id)
            if not any(
                (item.fragment_id, item.logical_core, item.fragment_record_index, item.operand_id) == key
                for item in address_bindings
            ):
                candidates = sorted(candidates, key=lambda item: (item.region_offset_bytes, item.id))
                address_bindings.append(AddressOperandBinding(
                    fragment.id,
                    stream.logical_core,
                    relocation.record_index,
                    relocation.operand_id,
                    tuple(item.id for item in candidates),
                    tuple(item.tensor_slice for item in candidates),
                ))
    program_definitions = []
    for symbol in fragment.program_symbols:
        cores = tuple(sorted(symbol_uses[symbol.id], key=lambda item: (item.die_id, item.local_core_id)))
        if symbol.kind is ProgramSymbolKind.SRAM_LABEL:
            name, value, size = f"unf_label_{symbol.id[-16:]}", 0, 0
        elif symbol.kind is ProgramSymbolKind.SRAM_REGION:
            root = next(
                item for item in fragment.buffer_abi
                if item.alias_of is None and item.region_ref == symbol.source_ref
                and item.logical_core in cores
            )
            binding = storage_binding[(actions[next(
                view.task_ref for view in operand_abi.operands
                if view.storage_ref == root_storage_ref[root.storage_id]
                and task_core[view.task_ref] == root.logical_core
            )].rank, root_storage_ref[root.storage_id])]
            region = _region(ir1, binding)
            name, value, size = region.name, region.base_bytes, region.size_bytes
        elif symbol.source_ref in buffer_by_binding:
            abi = buffer_by_binding[symbol.source_ref]
            root = next(item for item in fragment.buffer_abi if item.binding_id == abi.alias_of)
            binding = storage_binding[(actions[next(
                view.task_ref for view in operand_abi.operands if view.value_ref == abi.value_id
            )].rank, root_storage_ref[root.storage_id])]
            name = f"unf_abs_{symbol.id[-16:]}"
            value = binding.base_address + (abi.region_offset_bytes - root.region_offset_bytes)
            size = abi.size_bytes
        else:
            matching = [
                item for item in address_bindings
                if any(
                    stream.logical_core == item.logical_core
                    and relocation.record_index == item.fragment_record_index
                    and relocation.operand_id == item.operand_id
                    and relocation.symbol_ref == symbol.id
                    for stream in fragment.core_streams
                    for relocation in stream.address_relocations
                )
            ]
            if len(matching) != 1:
                raise SchemaError("reduce span requires one exact witness", path="fragment.program_symbols")
            abis = [buffer_by_id[ref] for ref in matching[0].buffer_abi_ids]
            starts = sorted((item.region_offset_bytes, item.size_bytes) for item in abis)
            root = next(item for item in fragment.buffer_abi if item.binding_id == abis[0].alias_of)
            action = actions[next(
                stream.records[matching[0].fragment_record_index].source_global_action_id
                for stream in fragment.core_streams
                if stream.logical_core == matching[0].logical_core
            )]
            binding = storage_binding[
                (action.rank, root_storage_ref[root.storage_id])
            ]
            value = binding.base_address + (starts[0][0] - root.region_offset_bytes)
            size = sum(item[1] for item in starts)
            name = f"unf_reduce_{symbol.id[-16:]}"
        program_definitions.append(ProgramSymbolDefinition(symbol, name, value, size, cores))
    runtime_defs = []
    flow_by_task = {
        task_ref: flow for flow in projection.flows
        for task_ref in (flow.send_task_ref, flow.recv_task_ref)
    }
    wait_by_recv = {
        dependency: action.id
        for action in actions.values()
        if action.kind is SwizzleActionKind.WAIT
        for dependency in action.deps
        if actions[dependency].kind is SwizzleActionKind.RECV
    }
    local_copy_token = {
        item.token_symbol_ref: item.task_ref
        for item in core_abi.runtime_bindings
        if actions[item.task_ref].kind is SwizzleActionKind.LOCAL_COPY
    }
    for symbol in fragment.runtime_symbols:
        if symbol.kind is RuntimeSymbolKind.DTE_FSM:
            flow = next(
                flow for flow in projection.flows
                if any(item.fsm_symbol_ref == symbol.id and item.flow_ref == flow.id for item in core_abi.runtime_bindings)
            )
            cores = tuple(sorted((task_core[flow.send_task_ref], task_core[flow.recv_task_ref]), key=lambda item: (item.die_id, item.local_core_id)))
            source, destination = flow.send_task_ref, flow.recv_task_ref
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            recv = next((item.task_ref for item in core_abi.runtime_bindings if item.token_symbol_ref == symbol.id and actions[item.task_ref].kind is SwizzleActionKind.RECV), None)
            owner = recv if recv is not None else local_copy_token[symbol.id]
            cores = (task_core[owner],)
            source, destination = owner, wait_by_recv.get(owner, owner)
        elif symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            event = next((item for item in core_abi.barrier_events if symbol.id in (item.source_core_symbol_ref, item.destination_core_symbol_ref)), None)
            if event is not None:
                represented = event.source_core if symbol.id == event.source_core_symbol_ref else event.destination_core
            else:
                represented = next(item.peer_core for item in core_abi.runtime_bindings if item.peer_symbol_ref == symbol.id)
            cores, source, destination = (represented,), None, None
        elif symbol.kind is RuntimeSymbolKind.EVENT_TAG:
            event = next(item for item in core_abi.barrier_events if item.event_symbol_ref == symbol.id)
            cores = tuple(sorted((event.source_core, event.destination_core), key=lambda item: (item.die_id, item.local_core_id)))
            source, destination = event.source_task_ref, event.destination_task_ref
        else:
            raise SchemaError("unexpected runtime symbol kind", path="fragment.runtime_symbols")
        runtime_defs.append(RuntimeSymbolDefinition(symbol, cores, source, destination))
    active_cores = tuple(stream.logical_core for stream in fragment.core_streams)
    starts = []
    for core in active_cores:
        first = min((item for item in core_abi.task_bindings if item.logical_core == core), key=lambda item: item.core_order)
        symbol = RuntimeSymbol(_id("start_tag", {"projection": projection.id, "core": core, "first": first.task_ref}), RuntimeSymbolKind.START_TAG, first.task_ref)
        runtime_defs.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        starts.append(LogicalStartEvent(core, symbol.id, 1))
    terminal_cores = tuple(sorted({task_core[item.terminal_task_ref] for item in projection.ranks}, key=lambda item: (item.die_id, item.local_core_id)))
    core_bindings = []
    linked_streams = []
    for stream in fragment.core_streams:
        task_binding = task_binding_by_core[stream.logical_core]
        die = next(item for item in ir1.fabric.dies if item.id == stream.logical_core.die_id)
        core = next(item for item in die.cores if item.local_core_id == stream.logical_core.local_core_id)
        core_bindings.append(CoreRuntimeBinding(stream.logical_core, core.id, task_binding.runtime_core_id, core.sram_profile_ref))
        linked_streams.append(LinkedCoreStream(stream.logical_core, task_binding.runtime_core_id, tuple(
            LinkedRecordRef(fragment.id, index, record.source_global_action_id)
            for index, record in enumerate(stream.records)
        )))
    entry_counts = defaultdict(int)
    exit_counts = defaultdict(int)
    for stream in fragment.core_streams:
        for record in stream.records:
            operands = {item.name: item for item in record.operands}
            if record.opcode is RecordOpcode.EVENT_WAIT:
                entry_counts[operands["tag"].symbol_ref] += operands["count"].literal_value
            elif record.opcode is RecordOpcode.EVENT_SET:
                exit_counts[operands["tag"].symbol_ref] += 1
    interface = FragmentInterface(
        fragment.id, (), tuple(sorted(item.id for item in fragment.runtime_symbols)),
        (), tuple(sorted(item.id for item in fragment.program_symbols)),
        tuple(EventCredit(key, entry_counts[key]) for key in sorted(entry_counts)),
        tuple(EventCredit(key, exit_counts[key]) for key in sorted(exit_counts)),
    )
    artifacts = (
        (ManifestInputKind.IR1, ir1),
        (ManifestInputKind.UNFUSED_COMPARISON_BASELINE, plan.baseline),
        (ManifestInputKind.UNFUSED_COMPARISON_PLAN, plan),
        (ManifestInputKind.UNFUSED_COMPARISON_PROJECTION, projection),
        (ManifestInputKind.UNFUSED_COMPARISON_LOWERED, lowered),
        (ManifestInputKind.UNFUSED_COMPARISON_CORE_ABI, core_abi),
        (ManifestInputKind.UNFUSED_COMPARISON_OPERAND_ABI, operand_abi),
        (ManifestInputKind.COMMAND_FRAGMENT, fragment),
    )
    digests = tuple(sorted(
        (ManifestInputDigest(kind, artifact.id, artifact.schema_version, canonical_digest(artifact)) for kind, artifact in artifacts),
        key=lambda item: (item.kind.value, item.artifact_id),
    ))
    result = LinkedProgramManifest.create(
        producer_pass="unfused_comparison_standard_linker",
        capabilities=0,
        source_ir1_id=ir1.id,
        source_projection_id=projection.id,
        source_schedule_set_id=core_abi.id,
        source_global_dag_id=projection.id,
        input_digests=digests,
        fragments=(fragment,),
        fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings),
        core_streams=tuple(linked_streams),
        runtime_symbol_definitions=tuple(sorted(runtime_defs, key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program_definitions, key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=(),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            active_cores, tuple(starts), terminal_cores, active_cores, terminal_cores,
            EmptyCoreAckPolicy.INCLUDE_EMPTY, ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    result.validate()
    return result


def link_unfused_comparison_program(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
    projection: UnfusedComparisonProjection,
    lowered: UnfusedComparisonLoweredProgram,
    core_abi: UnfusedComparisonCoreABI,
    operand_abi: UnfusedComparisonOperandABI,
) -> UnfusedComparisonStandardLinkedProgram:
    fragment = lower_unfused_comparison_fragment(
        ir1, plan, projection, lowered, core_abi, operand_abi
    )
    manifest = link_unfused_comparison_manifest(
        ir1, plan, projection, lowered, core_abi, operand_abi, fragment
    )
    result = UnfusedComparisonStandardLinkedProgram.create(
        ir1=ir1,
        plan=plan,
        projection=projection,
        lowered=lowered,
        core_abi=core_abi,
        operand_abi=operand_abi,
        fragment=fragment,
        manifest=manifest,
    )
    result.validate_against()
    return result


__all__ = [
    "link_unfused_comparison_manifest",
    "link_unfused_comparison_program",
    "lower_unfused_comparison_fragment",
]
