"""Canonical standard fragment emission for replacement-only MoE Swizzle."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreRuntimeBinding,
    CoreFragmentStream,
    EmptyCoreAckPolicy,
    FragmentInterface,
    FragmentKind,
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
    RuntimeRelocation,
    RuntimeSymbol,
    RuntimeSymbolDefinition,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.swizzle import SwizzleActionKind
from ..schema.swizzle_moe_abi import MoeSwizzleCoreAddressABI
from ..schema.swizzle_moe_ir2 import MoeSwizzleIr2Projection
from ..schema.swizzle_moe_operand_abi import MoeSwizzleOperandABI
from ..schema.swizzle_plan import SwizzleValueUse


_SCHEMA = "wafer_frontend.moe_swizzle_standard_lowering/v1alpha1"
_PRODUCER = "moe_swizzle_standard_lowering"


def _id(kind: str, semantic: object) -> str:
    return stable_artifact_id(f"moe_swizzle_standard_{kind}", semantic, schema_version=_SCHEMA)


def _region(ir1: IR1, binding: object) -> object:
    core = next(
        item
        for die in ir1.fabric.dies
        if die.id == binding.logical_core.die_id
        for item in die.cores
        if item.local_core_id == binding.logical_core.local_core_id
    )
    profile = next(item for item in ir1.fabric.sram_profiles if item.id == core.sram_profile_ref)
    return next(item for item in profile.regions if item.id == binding.region_ref)


def _relocations(records: list[RelocatableRecord]) -> tuple[tuple[RuntimeRelocation, ...], tuple[AddressRelocation, ...]]:
    runtime = []
    address = []
    for record_index, record in enumerate(records):
        for operand in record.operands:
            if operand.runtime_field is not None:
                runtime.append(RuntimeRelocation(record_index, operand.runtime_field, operand.symbol_ref))
            elif operand.operand_id is not None:
                bind_label = (
                    SemanticOperandId.SRAM_BIND_INPUT_0
                    <= operand.operand_id
                    <= SemanticOperandId.SRAM_BIND_OUTPUT
                )
                kind = ProgramSymbolKind.SRAM_LABEL if bind_label else {
                    SemanticOperandId.REGION_NAME: ProgramSymbolKind.SRAM_REGION,
                    SemanticOperandId.LABEL_SYMBOL: ProgramSymbolKind.SRAM_LABEL,
                    SemanticOperandId.SYMBOL: ProgramSymbolKind.SRAM_LABEL,
                }.get(operand.operand_id, ProgramSymbolKind.ABSOLUTE_ADDRESS)
                address.append(AddressRelocation(record_index, operand.operand_id, kind, operand.symbol_ref, 0))
    runtime.sort(key=lambda item: (item.record_index, list(RuntimeOperandField).index(item.field)))
    address.sort(key=lambda item: (item.record_index, int(item.operand_id)))
    return tuple(runtime), tuple(address)


def lower_moe_swizzle_standard_fragment(
    ir1: IR1,
    projection: MoeSwizzleIr2Projection,
    core_abi: MoeSwizzleCoreAddressABI,
    operand_abi: MoeSwizzleOperandABI,
) -> CommandFragment:
    ir1.validate("moe_standard.ir1")
    projection.validate("moe_standard.projection")
    core_abi.validate_against(ir1, projection, "moe_standard.core_abi")
    operand_abi.validate_against(projection, "moe_standard.operand_abi")
    tasks = {item.id: item for item in projection.tasks}
    values = {item.id: item for item in projection.values}
    task_bindings = {item.task_ref: item for item in core_abi.task_bindings}
    value_bindings = {item.value_ref: item for item in core_abi.value_bindings}
    runtime_bindings = {item.task_ref: item for item in core_abi.runtime_bindings}
    views_by_task = defaultdict(list)
    for view in operand_abi.operands:
        views_by_task[view.task_ref].append(view)
    matmuls = {item.task_ref: item for item in operand_abi.matmuls}
    dtes = {item.task_ref: item for item in operand_abi.dtes}
    swiglus = {item.task_ref: item for item in operand_abi.swiglus}

    program_symbols: dict[str, ProgramSymbol] = {}
    runtime_symbols: dict[str, RuntimeSymbol] = {}
    def program(kind: ProgramSymbolKind, source_ref: str) -> str:
        ref = _id("program_symbol", {"kind": kind, "source_ref": source_ref})
        program_symbols.setdefault(ref, ProgramSymbol(ref, kind, source_ref))
        return ref
    def runtime(kind: RuntimeSymbolKind, ref: str, source_ref: str) -> str:
        runtime_symbols.setdefault(ref, RuntimeSymbol(ref, kind, source_ref))
        return ref

    storage_orders = defaultdict(list)
    value_orders = defaultdict(list)
    for task in projection.tasks:
        binding = task_bindings[task.id]
        for view in views_by_task[task.id]:
            value_binding = value_bindings[view.value_ref]
            storage_orders[(binding.logical_core, value_binding.storage_ref)].append(binding.core_order)
            value_orders[view.value_ref].append(binding.core_order)

    buffers = []
    absolute = {}
    label = {}
    region_symbol = {}
    bindings_by_storage = defaultdict(list)
    for binding in value_bindings.values():
        bindings_by_storage[(binding.logical_core, binding.storage_ref)].append(binding)
    roots = {(item.logical_core, item.storage_ref): item for item in core_abi.storage_roots}
    element_bytes = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}
    for storage_key in sorted(bindings_by_storage, key=lambda item: (item[0].die_id, item[0].local_core_id, item[1])):
        members = sorted(bindings_by_storage[storage_key], key=lambda item: item.value_ref)
        root = roots[storage_key]
        member_values = [values[item.value_ref] for item in members]
        dtypes = {item.dtype for item in member_values}
        ownerships = {item.borrowed for item in member_values}
        if len(dtypes) != 1 or len(ownerships) != 1:
            raise SchemaError("packed storage root mixes dtype or ownership", path="moe_standard.buffer_abi")
        dtype = next(iter(dtypes))
        if dtype not in element_bytes or root.span_bytes % element_bytes[dtype]:
            raise SchemaError("packed storage root has an invalid dense extent", path="moe_standard.buffer_abi")
        orders = storage_orders[storage_key]
        if not orders:
            raise SchemaError("packed storage root has no operational use", path="moe_standard.buffer_abi")
        region = _region(ir1, root)
        root_binding_ref = _id("buffer_binding_root", {"core_abi": core_abi.id, "storage": root.storage_ref})
        root_semantic = {
            "schedule_id": core_abi.id,
            "binding_id": root_binding_ref,
            "value_id": root.storage_ref,
            "logical_core": root.logical_core,
            "tensor_slice": TensorSlice(root.storage_ref, (0,), (root.span_bytes // element_bytes[dtype],)),
            "region_ref": root.region_ref,
            "region_offset_bytes": root.address - region.base_bytes,
            "size_bytes": root.span_bytes,
            "alignment_bytes": 64,
            "banks": (),
            "storage_id": root.storage_ref,
            "alias_of": None,
            "lifetime_start": min(orders),
            "lifetime_end_exclusive": max(orders) + 1,
            "dtype": dtype,
            "layout": "swizzle_standard_storage_root/v1",
            "ownership": BufferOwnership.BORROWED if next(iter(ownerships)) else BufferOwnership.OWNED,
        }
        root_abi = BufferABI(id=_id("buffer_abi", root_semantic), **root_semantic)
        buffers.append(root_abi)
        label[root.storage_ref] = program(ProgramSymbolKind.SRAM_LABEL, root.storage_ref)
        region_symbol.setdefault(root.region_ref, program(ProgramSymbolKind.SRAM_REGION, root.region_ref))
        for binding in members:
            ref = binding.value_ref
            value = values[ref]
            orders = value_orders[ref]
            binding_ref = _id("buffer_binding", {"core_abi": core_abi.id, "value": ref, "slot": binding.slot})
            semantic = {
                "schedule_id": core_abi.id, "binding_id": binding_ref, "value_id": ref,
                "logical_core": binding.logical_core,
                "tensor_slice": TensorSlice(ref, (0,) * len(value.shape), value.shape),
                "region_ref": binding.region_ref,
                "region_offset_bytes": binding.address - region.base_bytes,
                "size_bytes": binding.size_bytes, "alignment_bytes": 64, "banks": (),
                "storage_id": binding.storage_ref, "alias_of": root_binding_ref,
                "lifetime_start": min(orders), "lifetime_end_exclusive": max(orders) + 1,
                "dtype": value.dtype, "layout": "swizzle_standard_storage_subview/v1",
                "ownership": BufferOwnership.ALIASED,
            }
            buffers.append(BufferABI(id=_id("buffer_abi", semantic), **semantic))
            absolute[ref] = program(ProgramSymbolKind.ABSOLUTE_ADDRESS, binding_ref)

    records_by_core = defaultdict(list)
    for task in sorted(projection.tasks, key=lambda item: (task_bindings[item.id].logical_core.die_id, task_bindings[item.id].logical_core.local_core_id, task_bindings[item.id].core_order)):
        owner = task.id
        binding = task_bindings[owner]
        records = records_by_core[binding.logical_core]
        views = sorted(views_by_task[owner], key=lambda item: item.ordinal)
        used = {value_bindings[item.value_ref].storage_ref for item in views}
        for storage_ref in sorted(used):
            root = next(item for item in core_abi.storage_roots if item.storage_ref == storage_ref and item.logical_core == binding.logical_core)
            if min(storage_orders[(binding.logical_core, storage_ref)]) == binding.core_order:
                region = _region(ir1, root)
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_ALLOC_AT, (
                    RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region_symbol[root.region_ref]),
                    RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label[storage_ref]),
                    RecordOperand.literal("region_offset_bytes", root.address - region.base_bytes),
                    RecordOperand.literal("size_bytes", root.span_bytes),
                    RecordOperand.literal("alignment_bytes", 64),
                    RecordOperand.literal("lifetime", 0),
                    RecordOperand.literal("spillable", region.spillable),
                )))
        if task.kind is SwizzleActionKind.COMP:
            contract = matmuls[owner]
            if len(views) != 3:
                raise SchemaError("MoE MATMUL requires two reads and one write", path="moe_standard.operands")
            lhs, rhs, output = views
            records.append(RelocatableRecord(owner, RecordOpcode.MATMUL, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, absolute[lhs.value_ref]),
                RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, absolute[rhs.value_ref]),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, absolute[output.value_ref]),
                RecordOperand.literal("parameters", (1, contract.m, contract.k, contract.n)),
            )))
        elif task.kind is SwizzleActionKind.SWIGLU:
            contract = swiglus[owner]
            if len(views) != 2:
                raise SchemaError("MoE SWIGLU requires one read and one write", path="moe_standard.operands")
            input_view, output_view = views
            input_storage = value_bindings[input_view.value_ref].storage_ref
            output_storage = value_bindings[output_view.value_ref].storage_ref
            records.append(RelocatableRecord(owner, RecordOpcode.SRAM_BIND, (
                RecordOperand.literal("input_count", 1),
                RecordOperand.address(
                    "input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0,
                    label[input_storage],
                ),
                *(
                    RecordOperand.literal(f"input_label_{index}", 0)
                    for index in range(1, 16)
                ),
                RecordOperand.address(
                    "output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                    label[output_storage],
                ),
            )))
            records.append(RelocatableRecord(owner, RecordOpcode.SWIGLU, (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address(
                    "input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                    absolute[input_view.value_ref],
                ),
                RecordOperand.literal("data_address", 0),
                RecordOperand.address(
                    "output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                    absolute[output_view.value_ref],
                ),
                RecordOperand.literal("parameters", (contract.element_count,)),
            )))
        elif task.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            contract = dtes[owner]
            rb = runtime_bindings[owner]
            assert rb.fsm_symbol_ref and rb.peer_core
            runtime(RuntimeSymbolKind.DTE_FSM, rb.fsm_symbol_ref, rb.flow_ref)
            peer_ref = _id("peer_core", {"task": owner, "peer": rb.peer_core})
            runtime(RuntimeSymbolKind.RUNTIME_CORE, peer_ref, str(rb.peer_core))
            common = (
                RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, peer_ref),
                RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0), RecordOperand.literal("epoch", 0),
            )
            view = views[0]
            if task.kind is SwizzleActionKind.SEND:
                opcode = RecordOpcode.DTE_SEND
                operands = (
                    RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0), RecordOperand.literal("completion", 1),
                    RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                    RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, rb.fsm_symbol_ref), RecordOperand.literal("token", 0),
                    RecordOperand.literal("length_bytes", contract.logical_bytes),
                    RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, absolute[view.value_ref]), *common,
                )
            else:
                assert rb.token_symbol_ref
                opcode = RecordOpcode.DTE_RECV
                runtime(RuntimeSymbolKind.DTE_TOKEN, rb.token_symbol_ref, owner)
                operands = (
                    RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0), RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
                    RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, rb.fsm_symbol_ref),
                    RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, rb.token_symbol_ref),
                    RecordOperand.literal("length_bytes", contract.logical_bytes),
                    RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, absolute[view.value_ref]), *common,
                )
            records.append(RelocatableRecord(owner, opcode, operands))
        elif task.kind is SwizzleActionKind.WAIT:
            rb = runtime_bindings[owner]
            assert rb.token_symbol_ref
            runtime(RuntimeSymbolKind.DTE_TOKEN, rb.token_symbol_ref, task.deps[0])
            records.append(RelocatableRecord(owner, RecordOpcode.DTE_WAIT, (
                RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, rb.token_symbol_ref),
            )))
        else:
            raise SchemaError("unsupported MoE standard task kind", path=f"moe_standard.tasks.{owner}")
        for storage_ref in sorted(used, reverse=True):
            if max(storage_orders[(binding.logical_core, storage_ref)]) == binding.core_order:
                records.append(RelocatableRecord(owner, RecordOpcode.SRAM_FREE, (
                    RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label[storage_ref]),
                )))

    streams = []
    for core, records in sorted(records_by_core.items(), key=lambda item: (item[0].die_id, item[0].local_core_id)):
        runtime_relocations, address_relocations = _relocations(records)
        streams.append(CoreFragmentStream(core, tuple(records), runtime_relocations, address_relocations))
    used_program = {item.symbol_ref for stream in streams for item in stream.address_relocations}
    result = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=projection.id,
        kind=FragmentKind.MOE_SWIZZLE,
        claimed_action_ids=tuple(sorted(tasks)),
        core_streams=tuple(streams),
        runtime_symbols=tuple(runtime_symbols[key] for key in sorted(runtime_symbols)),
        program_symbols=tuple(program_symbols[key] for key in sorted(used_program)),
        buffer_abi=tuple(sorted(buffers, key=lambda item: item.id)),
        state_abi=(),
    )
    result.validate()
    return result


__all__ = ["lower_moe_swizzle_standard_fragment"]
from ..schema.serde import canonical_digest
