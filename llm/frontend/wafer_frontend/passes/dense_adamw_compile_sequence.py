"""Bind real P3 AdamW operations to the DP1 physical WGRAD program."""

from __future__ import annotations

import re
import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, BufferABI, CommandFragment,
    CoreFragmentStream, FragmentInterface, LinkedCoreStream,
    LinkedProgramManifest, LinkedRecordRef, ManifestInputDigest,
    ManifestInputKind, OperandKind, ProgramSymbol, ProgramSymbolDefinition,
    ProgramSymbolKind, RecordOpcode, RecordOperand, RelocatableRecord,
    SemanticOperandId, StateABI, StateOperandBinding,
)
from ..schema.common import DType
from ..schema.dense_adamw_linked import DenseAdamwLinkedProgram
from ..schema.flexible_dense_backward import FlexibleDenseBackwardLinkedProgram
from ..schema.ir2 import TensorSlice
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import WorkloadOptimizerKind
from .flexible_dense_backward import (
    _align, _buffer, _hbm_symbol, _id, _label_symbol, _symbol,
)


_ROLE = (
    ("master", StateKind.OPTIMIZER_MASTER, DType.FP32),
    ("m", StateKind.OPTIMIZER_MOMENT1, DType.FP32),
    ("v", StateKind.OPTIMIZER_MOMENT2, DType.FP32),
    ("step", StateKind.OPTIMIZER_STEP, DType.INT32),
)
_INPUT_IDS = (
    SemanticOperandId.COMPUTE_INPUT_ADDRESS,
    SemanticOperandId.COMPUTE_DATA_ADDRESS,
    SemanticOperandId.COMPUTE_MASTER_ADDRESS,
    SemanticOperandId.COMPUTE_FIRST_MOMENT_ADDRESS,
    SemanticOperandId.COMPUTE_SECOND_MOMENT_ADDRESS,
    SemanticOperandId.COMPUTE_STEP_ADDRESS,
)
_OUTPUT_IDS = (
    SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
    SemanticOperandId.COMPUTE_UPDATED_MASTER_ADDRESS,
    SemanticOperandId.COMPUTE_UPDATED_FIRST_MOMENT_ADDRESS,
    SemanticOperandId.COMPUTE_UPDATED_SECOND_MOMENT_ADDRESS,
    SemanticOperandId.COMPUTE_UPDATED_STEP_ADDRESS,
)
_ADDRESS_NAMES = (
    "weight_address", "gradient_address", "master_weight_address",
    "first_moment_address", "second_moment_address",
    "step_counter_address", "updated_weight_address",
    "updated_master_weight_address", "updated_first_moment_address",
    "updated_second_moment_address", "updated_step_counter_address",
)


def _logical_names(tensor_ref: str) -> tuple[str, ...]:
    mapping = {
        "T0.final_norm.weight": ("final_norm.weight",),
        "T0.lm_head.weight": ("lm_head.weight",),
        "T0.tok_embeddings.weight": ("embedding.weight",),
    }
    if tensor_ref in mapping:
        return mapping[tensor_ref]
    matched = re.fullmatch(r"T0.layer([01]).w_(.+)", tensor_ref)
    if matched is None:
        raise SchemaError("unknown packed Dense physical weight", path=tensor_ref)
    layer, role = matched.groups()
    names = {
        "norm1": ("input_norm.weight",),
        "qkv": ("qkv.weight",),
        "o": ("attention_out.weight",),
        "norm2": ("post_norm.weight",),
        "gate_up": ("mlp_gate.weight", "mlp_up.weight"),
        "down": ("mlp_down.weight",),
    }.get(role)
    if names is None:
        raise SchemaError("unknown Dense parameter layout", path=tensor_ref)
    return tuple(f"layer.{layer}.{item}" for item in names)


def _optimizer_records(
    source: WorkloadMaterializationManifest,
    backward: FlexibleDenseBackwardLinkedProgram,
    step_index: int,
) -> LinkedProgramManifest:
    source.validate("source")
    backward.validate("backward")
    if (
        source.request.optimizer is None
        or source.request.optimizer.kind is not WorkloadOptimizerKind.ADAMW
        or step_index not in (0, 1)
    ):
        raise SchemaError("requires true step-0/1 AdamW workload", path="source")
    old = backward.manifest
    old_fragment = old.fragments[0]
    old_stream = old_fragment.core_streams[0]
    core = old_stream.logical_core
    if len(old.fragments) != 1 or len(old_fragment.core_streams) != 1:
        raise SchemaError("DP1 requires exactly one command stream", path="backward")
    space = backward.hbm_address_spaces[0]
    region = backward.fabric.sram_profiles[0].regions[0]
    old_buffers = {item.id: item for item in old_fragment.buffer_abi}
    old_state = {item.state_ref: item for item in old_fragment.state_abi}
    program_symbols: dict[str, ProgramSymbol] = {
        item.id: item for item in old_fragment.program_symbols
    }
    definitions: dict[str, ProgramSymbolDefinition] = {
        item.symbol.id: item for item in old.program_symbol_definitions
    }
    logical_versions = {
        item.logical_name: item
        for item in source.logical_graph.state_versions
        if item.version == 0
    }
    initial_values = {
        item.logical_name: item
        for item in source.logical_graph.tensor_values
        if item.state_ref in {state.id for state in logical_versions.values()}
    }
    operations = {
        (item.step, item.parameter_ref): item
        for item in source.logical_graph.operations
        if item.kind.value == "adamw_update"
    }
    buffer_by_source = {
        (item.source_global_action_id, item.opcode): item
        for item in old_stream.records
    }
    del buffer_by_source
    buffers: dict[tuple[str, str], BufferABI] = {}
    labels: dict[tuple[str, str], ProgramSymbol] = {}
    absolute: dict[tuple[str, str], ProgramSymbol] = {}
    for template in backward.plan.parameter_templates:
        for role in ("weight", "gradient"):
            matching = tuple(
                abi for abi in old_buffers.values()
                if abi.layout == f"flexible_dense_{role}_flat/v1"
                and abi.value_id == _id("value", {
                    "plan": backward.plan.id, "rank": core.die_id,
                    "state": template.state_ref, "role": role,
                })
            )
            if len(matching) != 1:
                raise SchemaError("physical WGRAD BufferABI does not close", path=role)
            abi = matching[0]
            buffers[(template.state_ref, role)] = abi
            absolute[(template.state_ref, role)] = next(
                symbol for symbol in program_symbols.values()
                if symbol.source_ref == abi.binding_id
                and symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
            )
            labels[(template.state_ref, role)] = next(
                symbol for symbol in program_symbols.values()
                if symbol.source_ref == abi.storage_id
                and symbol.kind is ProgramSymbolKind.SRAM_LABEL
            )
    hbm_cursor = _align(
        max(item.address + item.size_bytes for item in old_state.values()),
        space.alignment_bytes,
    )
    sram_cursor = _align(
        max(item.region_offset_bytes + item.size_bytes for item in old_buffers.values()),
        64,
    )
    optimizer_states: dict[tuple[str, str], StateABI] = {}
    optimizer_buffers: dict[tuple[str, str], BufferABI] = {}
    optimizer_hbm: dict[tuple[str, str], ProgramSymbol] = {}
    optimizer_labels: dict[tuple[str, str], ProgramSymbol] = {}
    optimizer_absolute: dict[tuple[str, str], ProgramSymbol] = {}
    seen_logical: set[str] = set()
    for template in backward.plan.parameter_templates:
        names = _logical_names(template.tensor_ref)
        value_sizes = tuple(
            initial_values[f"{name}.rank0"].size_bytes for name in names
        )
        if sum(value_sizes) != template.weight_bytes:
            raise SchemaError(
                "gate/up physical slices must cover one exact packed FP16 carrier",
                path=template.tensor_ref,
            )
        for name in names:
            if name in seen_logical or (step_index, name) not in operations:
                raise SchemaError("17 AdamW operations must bind uniquely", path=name)
            seen_logical.add(name)
            for role, kind, dtype in _ROLE:
                logical_ref = f"optimizer.adamw.{role}.{name}"
                version = logical_versions.get(logical_ref)
                value = initial_values.get(f"{logical_ref}.rank0")
                if version is None or value is None or version.kind.value != kind.value:
                    raise SchemaError(
                        "AdamW optimizer state must derive from real E2E graph",
                        path=logical_ref,
                    )
                expected = (4 if role == "step" else 2 * initial_values[f"{name}.rank0"].size_bytes)
                if value.size_bytes != expected or value.dtype is not dtype:
                    raise SchemaError(
                        "AdamW state FP32 bytes / INT32 step differ from P3",
                        path=logical_ref,
                    )
                hbm_cursor = _align(hbm_cursor, space.alignment_bytes)
                binding_ref = _id("adamw_hbm_binding", {
                    "source": source.id, "state": logical_ref,
                    "die": core.die_id, "address": hbm_cursor,
                })
                state = StateABI.create(
                    state_ref=logical_ref, hbm_binding_ref=binding_ref,
                    kind=kind, lifetime=PersistentStateLifetime.PERSISTENT,
                    access=PersistentStateAccess.READ_WRITE,
                    shape=value.shape, dtype=dtype,
                    layout=f"dense_adamw_{role}_flat/v1",
                    die_id=core.die_id, address=hbm_cursor,
                    size_bytes=value.size_bytes,
                    alignment_bytes=space.alignment_bytes,
                )
                optimizer_states[(name, role)] = state
                hbm_symbol = _hbm_symbol(binding_ref)
                optimizer_hbm[(name, role)] = hbm_symbol
                program_symbols[hbm_symbol.id] = hbm_symbol
                definitions[hbm_symbol.id] = ProgramSymbolDefinition(
                    hbm_symbol,
                    f"adamw_hbm_{role}_{name.replace('.', '_')}",
                    state.address, state.size_bytes, (core,),
                )
                hbm_cursor += state.size_bytes
                sram_cursor = _align(sram_cursor, 64)
                carrier = max(value.size_bytes, 256)
                buffer = _buffer(
                    plan=backward.plan, core=core, region_ref=region.id,
                    offset=sram_cursor, template=template,
                    role=f"adamw_{role}_{name.replace('.', '_')}",
                    dtype=dtype, size_bytes=carrier,
                    lifetime_end=len(backward.plan.rank_actions) + 1,
                )
                optimizer_buffers[(name, role)] = buffer
                abs_symbol, label_symbol = _symbol(buffer), _label_symbol(buffer)
                optimizer_absolute[(name, role)] = abs_symbol
                optimizer_labels[(name, role)] = label_symbol
                for symbol, name_tag, val, sz in (
                    (abs_symbol, "abs", region.base_bytes + sram_cursor, carrier),
                    (label_symbol, "label", 0, 0),
                ):
                    program_symbols[symbol.id] = symbol
                    definitions[symbol.id] = ProgramSymbolDefinition(
                        symbol, f"adamw_{name_tag}_{role}_{name.replace('.', '_')}",
                        val, sz, (core,),
                    )
                sram_cursor += carrier
    if (
        len(seen_logical) != 17 or len(optimizer_states) != 68
        or hbm_cursor > space.base_address + space.size_bytes
        or region.base_bytes + sram_cursor > region.base_bytes + region.size_bytes
        or region.base_bytes + sram_cursor - 1 > 0xffff
    ):
        raise SchemaError(
            "DP1 AdamW source/state capacity or uint16 compute address exceeds hardware",
            path="fabric",
        )

    records: list[RelocatableRecord] = []
    relocations: list[AddressRelocation] = []
    address_closures: list[tuple[int, SemanticOperandId, BufferABI, TensorSlice]] = []
    state_closures: list[tuple[int, StateABI]] = []
    old_relocs: dict[int, list[AddressRelocation]] = {}
    old_addr_closures: dict[int, list[tuple[SemanticOperandId, BufferABI, TensorSlice]]] = {}
    old_state_closures: dict[int, list[StateABI]] = {}
    for relocation in old_stream.address_relocations:
        old_relocs.setdefault(relocation.record_index, []).append(relocation)
    for binding in old.address_operand_bindings:
        if len(binding.buffer_abi_ids) != 1:
            raise SchemaError("DP1 WGRAD requires one BufferABI per address", path="backward")
        old_addr_closures.setdefault(binding.fragment_record_index, []).append(
            (binding.operand_id, old_buffers[binding.buffer_abi_ids[0]],
             binding.tensor_slices[0])
        )
    for binding in old.state_operand_bindings:
        abi = next(item for item in old_state.values() if item.id == binding.state_abi_id)
        old_state_closures.setdefault(binding.fragment_record_index, []).append(abi)

    def push(
        record: RelocatableRecord,
        accesses: tuple[tuple[SemanticOperandId, ProgramSymbol, BufferABI | StateABI,
                              ProgramSymbolKind, int], ...] = (),
    ) -> None:
        idx = len(records)
        records.append(record)
        for operand, symbol, abi, kind, addend in accesses:
            relocations.append(AddressRelocation(
                idx, operand, kind, symbol.id, addend,
            ))
            if type(abi) is StateABI:
                state_closures.append((idx, abi))
            else:
                if addend and record.opcode is RecordOpcode.ADAMW_UPDATE:
                    dtype_bytes = 4 if abi.dtype is DType.FP32 else 2
                    count = record.operands[16].literal_value
                    view = TensorSlice(
                        abi.value_id, (addend // dtype_bytes,), (count,),
                    )
                else:
                    view = abi.tensor_slice
                address_closures.append((idx, operand, abi, view))

    def copy_old(index: int) -> None:
        record = old_stream.records[index]
        idx = len(records)
        records.append(record)
        for relocation in old_relocs.get(index, ()):
            relocations.append(AddressRelocation(
                idx, relocation.operand_id, relocation.symbol_kind,
                relocation.symbol_ref, relocation.addend,
            ))
        address_closures.extend(
            (idx, operand, abi, view)
            for operand, abi, view in old_addr_closures.get(index, ())
        )
        state_closures.extend(
            (idx, state) for state in old_state_closures.get(index, ())
        )

    region_symbol = next(
        symbol for symbol in program_symbols.values()
        if symbol.kind is ProgramSymbolKind.SRAM_REGION
        and symbol.source_ref == region.id
    )

    def insert_optimizer(name: str, template, weight_offset: int) -> None:
        action = operations[(step_index, name)].id
        weight = buffers[(template.state_ref, "weight")]
        gradient = buffers[(template.state_ref, "gradient")]
        weight_sym = absolute[(template.state_ref, "weight")]
        gradient_sym = absolute[(template.state_ref, "gradient")]
        weight_label = labels[(template.state_ref, "weight")]
        gradient_label = labels[(template.state_ref, "gradient")]
        for role, _kind, _dtype in _ROLE:
            load_action = f"{action}.optimizer_{role}_load"
            abi = optimizer_buffers[(name, role)]
            label = optimizer_labels[(name, role)]
            push(
                RelocatableRecord(load_action, RecordOpcode.SRAM_ALLOC_AT, (
                    RecordOperand.address(
                        "region_name", SemanticOperandId.REGION_NAME,
                        region_symbol.id,
                    ),
                    RecordOperand.address(
                        "label_symbol", SemanticOperandId.LABEL_SYMBOL,
                        label.id,
                    ),
                    RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
                    RecordOperand.literal("size_bytes", abi.size_bytes),
                    RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
                    RecordOperand.literal("lifetime", 0),
                    RecordOperand.literal("spillable", False),
                )),
                (
                    (SemanticOperandId.REGION_NAME, region_symbol, abi,
                     ProgramSymbolKind.SRAM_REGION, 0),
                    (SemanticOperandId.LABEL_SYMBOL, label, abi,
                     ProgramSymbolKind.SRAM_LABEL, 0),
                ),
            )
            state = optimizer_states[(name, role)]
            hbm_sym = optimizer_hbm[(name, role)]
            sram_sym = optimizer_absolute[(name, role)]
            push(
                RelocatableRecord(load_action, RecordOpcode.LSU_LOAD, (
                    RecordOperand.address(
                        "hbm_address", SemanticOperandId.HBM_ADDRESS, hbm_sym.id,
                    ),
                    RecordOperand.literal("size_bytes", state.size_bytes),
                    RecordOperand.address(
                        "destination_address", SemanticOperandId.DESTINATION_ADDRESS,
                        sram_sym.id,
                    ),
                )),
                (
                    (SemanticOperandId.HBM_ADDRESS, hbm_sym, state,
                     ProgramSymbolKind.ABSOLUTE_ADDRESS, 0),
                    (SemanticOperandId.DESTINATION_ADDRESS, sram_sym, abi,
                     ProgramSymbolKind.ABSOLUTE_ADDRESS, 0),
                ),
            )
        in_symbols = (
            weight_label, gradient_label,
            *(optimizer_labels[(name, role)] for role, _, _ in _ROLE),
        )
        in_abis = (
            weight, gradient,
            *(optimizer_buffers[(name, role)] for role, _, _ in _ROLE),
        )
        bind_operands = [RecordOperand.literal("input_count", 6)]
        for slot in range(16):
            operand = SemanticOperandId(
                int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot,
            )
            bind_operands.append(
                RecordOperand.address(f"input_label_{slot}", operand,
                                      in_symbols[slot].id)
                if slot < 6
                else RecordOperand.literal(f"input_label_{slot}", 0)
            )
        bind_operands.append(RecordOperand.address(
            "output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
            weight_label.id,
        ))
        push(
            RelocatableRecord(action, RecordOpcode.SRAM_BIND, tuple(bind_operands)),
            (
                *((
                    SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + slot),
                    symbol, abi, ProgramSymbolKind.SRAM_LABEL, 0,
                ) for slot, (symbol, abi) in enumerate(zip(in_symbols, in_abis))),
                (SemanticOperandId.SRAM_BIND_OUTPUT, weight_label, weight,
                 ProgramSymbolKind.SRAM_LABEL, 0),
            ),
        )
        address_symbols = (
            weight_sym, gradient_sym,
            *(optimizer_absolute[(name, role)] for role, _, _ in _ROLE),
            weight_sym, *(optimizer_absolute[(name, role)] for role, _, _ in _ROLE),
        )
        address_abis = (
            weight, gradient,
            *(optimizer_buffers[(name, role)] for role, _, _ in _ROLE),
            weight, *(optimizer_buffers[(name, role)] for role, _, _ in _ROLE),
        )
        address_addends = (
            weight_offset, 2 * weight_offset, 0, 0, 0, 0,
            weight_offset, 0, 0, 0, 0,
        )
        weight_bytes = initial_values[f"{name}.rank0"].size_bytes
        opts = source.request.optimizer
        assert opts is not None and opts.beta1 is not None
        assert opts.beta2 is not None and opts.epsilon is not None
        compute_operands = [
            RecordOperand.literal("weight_datatype", 1),
            RecordOperand.literal("gradient_datatype", 3),
            RecordOperand.literal("state_datatype", 3),
            RecordOperand.literal("output_datatype", 1),
            RecordOperand.literal("rounding", 0),
        ]
        for field, operand, symbol in zip(
            _ADDRESS_NAMES, (*_INPUT_IDS, *_OUTPUT_IDS), address_symbols,
        ):
            compute_operands.append(RecordOperand.address(
                field, operand, symbol.id,
            ))
        compute_operands.extend((
            RecordOperand.literal("element_count", weight_bytes // 2),
            RecordOperand.literal("step", step_index + 1),
        ))
        for field in (
            "learning_rate", "beta1", "beta2", "epsilon", "weight_decay",
        ):
            compute_operands.append(RecordOperand.literal(
                f"{field}_f64_bits",
                struct.unpack("<Q", struct.pack("<d", getattr(opts, field)))[0],
            ))
        push(
            RelocatableRecord(action, RecordOpcode.ADAMW_UPDATE,
                              tuple(compute_operands)),
            tuple(
                (operand, symbol, abi, ProgramSymbolKind.ABSOLUTE_ADDRESS, addend)
                for operand, symbol, abi, addend in zip(
                    (*_INPUT_IDS, *_OUTPUT_IDS),
                    address_symbols, address_abis, address_addends,
                )
            ),
        )
        for role, _kind, _dtype in _ROLE:
            store_action = f"{action}.optimizer_{role}_store"
            state = optimizer_states[(name, role)]
            hbm_sym = optimizer_hbm[(name, role)]
            sram_sym = optimizer_absolute[(name, role)]
            abi = optimizer_buffers[(name, role)]
            push(
                RelocatableRecord(store_action, RecordOpcode.LSU_STORE, (
                    RecordOperand.address(
                        "hbm_address", SemanticOperandId.HBM_ADDRESS, hbm_sym.id,
                    ),
                    RecordOperand.literal("size_bytes", state.size_bytes),
                    RecordOperand.address(
                        "source_address", SemanticOperandId.SOURCE_ADDRESS,
                        sram_sym.id,
                    ),
                )),
                (
                    (SemanticOperandId.HBM_ADDRESS, hbm_sym, state,
                     ProgramSymbolKind.ABSOLUTE_ADDRESS, 0),
                    (SemanticOperandId.SOURCE_ADDRESS, sram_sym, abi,
                     ProgramSymbolKind.ABSOLUTE_ADDRESS, 0),
                ),
            )
            label = optimizer_labels[(name, role)]
            push(
                RelocatableRecord(store_action, RecordOpcode.SRAM_FREE, (
                    RecordOperand.address(
                        "symbol", SemanticOperandId.SYMBOL, label.id,
                    ),)),
                ((SemanticOperandId.SYMBOL, label, abi,
                  ProgramSymbolKind.SRAM_LABEL, 0),),
            )

    action_template = {
        item.id: next(t for t in backward.plan.parameter_templates
                      if t.state_ref == item.state_ref)
        for item in backward.plan.rank_actions if item.state_ref is not None
    }
    for index, old_record in enumerate(old_stream.records):
        if old_record.opcode is RecordOpcode.SRAM_BIND and (
            index + 1 < len(old_stream.records)
            and old_stream.records[index + 1].opcode is RecordOpcode.SGD_UPDATE
            and old_stream.records[index + 1].source_global_action_id
                == old_record.source_global_action_id
        ):
            template = action_template[old_record.source_global_action_id]
            offset = 0
            for logical_name in _logical_names(template.tensor_ref):
                insert_optimizer(logical_name, template, offset)
                offset += initial_values[f"{logical_name}.rank0"].size_bytes
            if offset != template.weight_bytes:
                raise SchemaError("gate/up weight slices overlap or leave a gap", path=template.tensor_ref)
            continue
        if old_record.opcode is RecordOpcode.SGD_UPDATE:
            continue
        copy_old(index)
    if len(tuple(r for r in records if r.opcode is RecordOpcode.ADAMW_UPDATE)) != 17:
        raise SchemaError("must bind all 17 logical AdamW source operations", path="source")
    fragment = CommandFragment.create(
        producer_pass="dense_adamw_lowering",
        source_global_dag_id=old.source_global_dag_id,
        kind=old_fragment.kind,
        claimed_action_ids=tuple(sorted({
            record.source_global_action_id for record in records
        })),
        core_streams=(CoreFragmentStream(
            core, tuple(records), (),
            tuple(sorted(
                relocations,
                key=lambda item: (item.record_index, int(item.operand_id)),
            )),
        ),),
        runtime_symbols=(),
        program_symbols=tuple(sorted(program_symbols.values(), key=lambda item: item.id)),
        buffer_abi=tuple(sorted(
            (*old_buffers.values(), *optimizer_buffers.values()),
            key=lambda item: item.id,
        )),
        state_abi=tuple(sorted(
            (*old_state.values(), *optimizer_states.values()),
            key=lambda item: item.id,
        )),
    )
    fragment.validate("dense_adamw.fragment")
    fragments = (fragment,)
    digests = tuple(sorted(
        (
            *(
                item for item in old.input_digests
                if item.kind is not ManifestInputKind.COMMAND_FRAGMENT
            ),
            ManifestInputDigest(
                ManifestInputKind.COMMAND_FRAGMENT,
                fragment.id, fragment.schema_version, canonical_digest(fragment),
            ),
            ManifestInputDigest(
                ManifestInputKind.DENSE_ADAMW_SOURCE,
                source.id, source.schema_version, canonical_digest(source),
            ),
        ),
        key=lambda item: (item.kind.value, item.artifact_id),
    ))
    manifest = LinkedProgramManifest.create(
        producer_pass="dense_adamw_linker",
        capabilities=old.capabilities,
        source_ir1_id=old.source_ir1_id,
        source_projection_id=old.source_projection_id,
        source_schedule_set_id=old.source_schedule_set_id,
        source_global_dag_id=old.source_global_dag_id,
        input_digests=digests,
        fragments=fragments,
        fragment_interfaces=(FragmentInterface(
            fragment.id, (), (), (),
            tuple(item.id for item in fragment.program_symbols), (), (),
        ),),
        core_bindings=old.core_bindings,
        core_streams=(LinkedCoreStream(
            core, old.core_streams[0].runtime_core_id,
            tuple(LinkedRecordRef(
                fragment.id, index, record.source_global_action_id,
            ) for index, record in enumerate(records)),
        ),),
        runtime_symbol_definitions=(),
        program_symbol_definitions=tuple(sorted(
            definitions.values(), key=lambda item: item.symbol.id,
        )),
        address_operand_bindings=tuple(sorted(
            (
                AddressOperandBinding(
                    fragment.id, core, index, operand,
                    (abi.id,), (view,),
                )
                for index, operand, abi, view in address_closures
            ),
            key=lambda item: (
                item.logical_core.die_id, item.logical_core.local_core_id,
                item.fragment_id, item.fragment_record_index, int(item.operand_id),
            ),
        )),
        state_operand_bindings=tuple(sorted(
            (
                StateOperandBinding(
                    fragment.id, core, index,
                    SemanticOperandId.HBM_ADDRESS, abi.id,
                )
                for index, abi in state_closures
            ),
            key=lambda item: (
                item.logical_core.die_id, item.logical_core.local_core_id,
                item.fragment_id, item.fragment_record_index, int(item.operand_id),
            ),
        )),
        core_groups=(),
        envelope=old.envelope,
    )
    manifest.validate("dense_adamw.manifest")
    return manifest


def compile_dense_adamw_step(
    source: WorkloadMaterializationManifest,
    backward: FlexibleDenseBackwardLinkedProgram,
    step_index: int,
) -> DenseAdamwLinkedProgram:
    """Lower one actual P3 AdamW step on the real 1x1 WGRAD physical carrier."""
    return DenseAdamwLinkedProgram.create(
        materialization=source,
        backward_source=backward,
        manifest=_optimizer_records(source, backward, step_index),
        step_index=step_index,
    )


__all__ = ["compile_dense_adamw_step"]
