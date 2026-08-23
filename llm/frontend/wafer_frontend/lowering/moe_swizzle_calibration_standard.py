"""Production standard lowering for one isolated MoE calibration primitive."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    CoreRuntimeBinding,
    EmptyCoreAckPolicy,
    EventCredit,
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
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.serde import canonical_digest
from ..schema.swizzle_moe_calibration import MoeCalibrationKind
from ..schema.swizzle_moe_calibration_program import (
    MoeSwizzleCalibrationProgramSource,
    MoeSwizzleCalibrationRecordCount,
    MoeSwizzleCalibrationStandardLinkedProgram,
    MoeSwizzleCalibrationTargetRecord,
    expected_moe_swizzle_calibration_record_quotient,
    required_moe_swizzle_calibration_target_opcode,
)
from ..schema.swizzle_moe_standard import MoeSwizzleStandardLinkedProgram


_LOWERING = "moe_swizzle_calibration_standard_lowering"
_LINKER = "moe_swizzle_calibration_standard_linker"
_SCHEMA = "wafer_frontend.moe_swizzle_calibration_standard_lowering/v1alpha1"
_validated_sources: set[tuple[str, str]] = set()
_validation_lock = Lock()


def _id(prefix: str, semantic: object) -> str:
    return stable_artifact_id(prefix, semantic, schema_version=_SCHEMA)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True, slots=True)
class _Recipe:
    recipe_id: str
    target_core: LogicalCoreRef
    target_runtime_core_id: int
    peer_core: LogicalCoreRef | None
    cores: tuple[LogicalCoreRef, ...]
    records: tuple[tuple[LogicalCoreRef, RelocatableRecord], ...]
    runtime_symbols: tuple[RuntimeSymbol, ...]
    runtime_definitions: tuple[RuntimeSymbolDefinition, ...]
    program_symbols: tuple[ProgramSymbol, ...]
    program_definitions: tuple[ProgramSymbolDefinition, ...]
    buffer_abi: tuple[BufferABI, ...]
    input_abi_ids: tuple[str, ...]
    output_abi_ids: tuple[str, ...]
    route_resource_refs: tuple[str, ...]


def _validate_source_once(source: MoeSwizzleStandardLinkedProgram) -> None:
    if type(source) is not MoeSwizzleStandardLinkedProgram:
        raise SchemaError(
            "source must be an exact whole MoE standard program", path="source"
        )
    digest = canonical_digest(source)
    key = (source.id, digest)
    with _validation_lock:
        if key in _validated_sources:
            return
    source.validate("source")
    with _validation_lock:
        _validated_sources.add(key)


def _core_fact(source: MoeSwizzleStandardLinkedProgram, core: LogicalCoreRef):
    return next(
        fact
        for die in source.hardware_facts.ordered_cores_by_die
        for fact in die
        if fact.logical_core == core
    )


def _source_core_binding(
    source: MoeSwizzleStandardLinkedProgram, core: LogicalCoreRef
) -> CoreRuntimeBinding:
    matches = tuple(
        item for item in source.manifest.core_bindings if item.logical_core == core
    )
    if len(matches) != 1:
        raise SchemaError(
            "calibration core lacks one exact source runtime binding",
            path="moe_swizzle_calibration.source.manifest.core_bindings",
        )
    return matches[0]


def _choose_cores(
    source: MoeSwizzleStandardLinkedProgram, kind: MoeCalibrationKind
) -> tuple[LogicalCoreRef, LogicalCoreRef | None, tuple[str, ...]]:
    transport = kind in {
        MoeCalibrationKind.DTE_LAUNCH,
        MoeCalibrationKind.DTE_SYNC,
        MoeCalibrationKind.DTE_HOP,
        MoeCalibrationKind.SESSION_OPEN,
        MoeCalibrationKind.SESSION_RETIRE,
    }
    two_core = transport or kind in {
        MoeCalibrationKind.EVENT_SET,
        MoeCalibrationKind.EVENT_WAIT,
    }
    if transport:
        if not source.hardware_facts.route_resources:
            raise SchemaError(
                "transport calibration requires one physical route resource",
                path="moe_swizzle_calibration.source.hardware_facts",
            )
        route = source.hardware_facts.route_resources[0]
        sender = source.hardware_facts.ordered_cores_by_die[route.source_die][0]
        receiver = source.hardware_facts.ordered_cores_by_die[
            route.destination_die
        ][0]
        if kind in {
            MoeCalibrationKind.DTE_LAUNCH,
            MoeCalibrationKind.DTE_SYNC,
            MoeCalibrationKind.SESSION_OPEN,
            MoeCalibrationKind.SESSION_RETIRE,
        }:
            return receiver.logical_core, sender.logical_core, (route.resource_id,)
        return sender.logical_core, receiver.logical_core, (route.resource_id,)
    ordered = tuple(
        binding.logical_core for binding in source.manifest.core_bindings
    )
    if not ordered:
        raise SchemaError("source has no active cores", path="source.manifest")
    target = ordered[0]
    if not two_core:
        return target, None, ()
    peer = next((core for core in ordered if core != target), None)
    if peer is None:
        raise SchemaError(
            "event calibration requires two real source cores",
            path="source.manifest.core_bindings",
        )
    if kind is MoeCalibrationKind.EVENT_WAIT:
        return peer, target, ()
    return target, peer, ()


def _shape_bytes(
    kind: MoeCalibrationKind, shape: tuple[int, int, int] | None
) -> tuple[tuple[int, ...], int]:
    if kind is MoeCalibrationKind.GROUP_GEMM:
        assert shape is not None
        m, n, k = shape
        return (m * k * 2, k * n * 2), m * n * 2
    if kind is MoeCalibrationKind.SWIGLU_GROUP:
        assert shape is not None
        _m, _intermediate, flattened = shape
        return (flattened * 4,), flattened * 2
    if kind is MoeCalibrationKind.SRAM_BIND:
        return (128,), 64
    return (64,), 64


def _make_buffer(
    *,
    recipe_id: str,
    name: str,
    core: LogicalCoreRef,
    fact: object,
    offset: int,
    size: int,
    ownership: BufferOwnership,
    lifetime_end: int,
) -> BufferABI:
    binding_id = _id("moe_swizzle_calibration_buffer", (recipe_id, name))
    storage_id = _id("moe_swizzle_calibration_storage", (recipe_id, name))
    value_id = _id("moe_swizzle_calibration_value", (recipe_id, name))
    semantic = {
        "schedule_id": recipe_id,
        "binding_id": binding_id,
        "value_id": value_id,
        "logical_core": core,
        "tensor_slice": TensorSlice(value_id, (0,), (size // 2,)),
        "region_ref": fact.region_name,
        "region_offset_bytes": fact.region_base_bytes + offset,
        "size_bytes": size,
        "alignment_bytes": fact.allocation_alignment_bytes,
        "banks": tuple(range(fact.bank_count)),
        "storage_id": storage_id,
        "alias_of": None,
        "lifetime_start": 0,
        "lifetime_end_exclusive": lifetime_end,
        "dtype": DType.FP16,
        "layout": f"moe_swizzle_calibration_{name}_root/v1",
        "ownership": ownership,
    }
    return BufferABI(
        id=stable_artifact_id("buffer_abi", semantic, schema_version=_SCHEMA),
        **semantic,
    )


def _build_recipe(
    source: MoeSwizzleStandardLinkedProgram,
    kind: MoeCalibrationKind,
    shape: tuple[int, int, int] | None,
) -> _Recipe:
    target, peer, routes = _choose_cores(source, kind)
    target_binding = _source_core_binding(source, target)
    recipe_id = _id(
        "moe_swizzle_calibration_recipe",
        (source.id, kind, shape, target, target_binding.runtime_core_id, routes),
    )
    sender = target
    receiver = peer
    if kind in {
        MoeCalibrationKind.DTE_LAUNCH,
        MoeCalibrationKind.DTE_SYNC,
        MoeCalibrationKind.SESSION_OPEN,
        MoeCalibrationKind.SESSION_RETIRE,
    }:
        assert peer is not None
        sender, receiver = peer, target
    elif kind is MoeCalibrationKind.DTE_HOP:
        assert peer is not None
        sender, receiver = target, peer

    input_core = sender if routes else target
    output_core = receiver if routes else target
    input_bytes, output_bytes = _shape_bytes(kind, shape)
    abis: list[BufferABI] = []
    offset_by_core: dict[LogicalCoreRef, int] = defaultdict(int)
    for index, size in enumerate(input_bytes):
        fact = _core_fact(source, input_core)
        offset = _align(
            offset_by_core[input_core], fact.allocation_alignment_bytes
        )
        abis.append(
            _make_buffer(
                recipe_id=recipe_id,
                name=f"input_{index}",
                core=input_core,
                fact=fact,
                offset=offset,
                size=size,
                ownership=BufferOwnership.BORROWED,
                lifetime_end=16,
            )
        )
        offset_by_core[input_core] = offset + size
    inputs = tuple(abis)
    fact = _core_fact(source, output_core)
    output_offset = _align(
        offset_by_core[output_core], fact.allocation_alignment_bytes
    )
    output = _make_buffer(
        recipe_id=recipe_id,
        name="output",
        core=output_core,
        fact=fact,
        offset=output_offset,
        size=output_bytes,
        ownership=BufferOwnership.OWNED,
        lifetime_end=16,
    )
    abis.append(output)
    offset_by_core[output_core] = output_offset + output_bytes
    scratch = None
    if kind is MoeCalibrationKind.SRAM_FREE:
        scratch_offset = _align(
            offset_by_core[output_core], fact.allocation_alignment_bytes
        )
        scratch = _make_buffer(
            recipe_id=recipe_id,
            name="scratch",
            core=output_core,
            fact=fact,
            offset=scratch_offset,
            size=output_bytes,
            ownership=BufferOwnership.OWNED,
            lifetime_end=16,
        )
        abis.append(scratch)

    program_symbols: dict[str, ProgramSymbol] = {}
    program_definitions: dict[str, ProgramSymbolDefinition] = {}

    def program(
        kind_: ProgramSymbolKind,
        source_ref: str,
        core: LogicalCoreRef,
        value: int,
        size: int,
        label: str,
    ) -> str:
        symbol = ProgramSymbol(
            _id("moe_swizzle_calibration_program_symbol", (recipe_id, kind_, source_ref, core)),
            kind_,
            source_ref,
        )
        program_symbols.setdefault(symbol.id, symbol)
        program_definitions.setdefault(
            symbol.id,
            ProgramSymbolDefinition(symbol, label, value, size, (core,)),
        )
        return symbol.id

    abs_by_abi = {
        abi.id: program(
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            abi.binding_id,
            abi.logical_core,
            abi.region_offset_bytes,
            abi.size_bytes,
            f"moe_cal_abs_{abi.id[-16:]}",
        )
        for abi in abis
    }
    regions_by_core = {}
    region_groups: dict[
        tuple[str, int, int], list[LogicalCoreRef]
    ] = defaultdict(list)
    for core in sorted(
        {abi.logical_core for abi in abis},
        key=lambda item: (item.die_id, item.local_core_id),
    ):
        core_fact = _core_fact(source, core)
        region_groups[
            (
                core_fact.region_name,
                core_fact.region_base_bytes,
                core_fact.region_size_bytes,
            )
        ].append(core)
    names = [key[0] for key in region_groups]
    if len(names) != len(set(names)):
        raise SchemaError(
            "one hardware SRAM region name must have one exact base/size",
            path="moe_swizzle_calibration.recipe.regions",
        )
    for (name, base, size), grouped in sorted(region_groups.items()):
        cores = tuple(grouped)
        source_key: object = cores[0] if len(cores) == 1 else cores
        symbol = ProgramSymbol(
            _id(
                "moe_swizzle_calibration_program_symbol",
                (recipe_id, ProgramSymbolKind.SRAM_REGION, name, source_key),
            ),
            ProgramSymbolKind.SRAM_REGION,
            name,
        )
        program_symbols[symbol.id] = symbol
        program_definitions[symbol.id] = ProgramSymbolDefinition(
            symbol, name, base, size, cores,
        )
        for core in cores:
            regions_by_core[core] = symbol.id
    label_by_abi = {
        abi.id: program(
            ProgramSymbolKind.SRAM_LABEL,
            abi.storage_id,
            abi.logical_core,
            0,
            0,
            f"moe_cal_label_{abi.id[-16:]}",
        )
        for abi in abis
    }

    runtime_symbols: dict[str, RuntimeSymbol] = {}
    runtime_definitions: dict[str, RuntimeSymbolDefinition] = {}

    def runtime(
        kind_: RuntimeSymbolKind,
        name: str,
        source_ref: str,
        cores: tuple[LogicalCoreRef, ...],
        source_action: str | None,
        destination_action: str | None,
    ) -> str:
        symbol = RuntimeSymbol(
            _id("moe_swizzle_calibration_runtime_symbol", (recipe_id, kind_, name)),
            kind_,
            source_ref,
        )
        runtime_symbols.setdefault(symbol.id, symbol)
        runtime_definitions.setdefault(
            symbol.id,
            RuntimeSymbolDefinition(
                symbol,
                tuple(sorted(cores, key=lambda item: (item.die_id, item.local_core_id))),
                source_action,
                destination_action,
            ),
        )
        return symbol.id

    owner = _id("moe_swizzle_calibration_action", (recipe_id, "target"))
    auxiliary = _id("moe_swizzle_calibration_action", (recipe_id, "auxiliary"))
    records: list[tuple[LogicalCoreRef, RelocatableRecord]] = []

    def emit(core: LogicalCoreRef, action: str, opcode: RecordOpcode, operands):
        records.append((core, RelocatableRecord(action, opcode, tuple(operands))))

    def bind_operands(
        input_abis: tuple[BufferABI, ...], output_abi: BufferABI
    ) -> tuple[RecordOperand, ...]:
        if not input_abis or len(input_abis) > 16:
            raise SchemaError(
                "SRAM_BIND input arity must be in [1,16]",
                path="moe_swizzle_calibration.recipe.bind",
            )
        return (
            RecordOperand.literal("input_count", len(input_abis)),
            *(
                RecordOperand.address(
                    f"input_label_{index}",
                    SemanticOperandId(
                        int(SemanticOperandId.SRAM_BIND_INPUT_0) + index
                    ),
                    label_by_abi[abi.id],
                )
                for index, abi in enumerate(input_abis)
            ),
            *(
                RecordOperand.literal(f"input_label_{index}", 0)
                for index in range(len(input_abis), 16)
            ),
            RecordOperand.address(
                "output_label",
                SemanticOperandId.SRAM_BIND_OUTPUT,
                label_by_abi[output_abi.id],
            ),
        )

    def alloc_operands(abi: BufferABI) -> tuple[RecordOperand, ...]:
        abi_fact = _core_fact(source, abi.logical_core)
        return (
            RecordOperand.address(
                "region_name",
                SemanticOperandId.REGION_NAME,
                regions_by_core[abi.logical_core],
            ),
            RecordOperand.address(
                "label_symbol",
                SemanticOperandId.LABEL_SYMBOL,
                label_by_abi[abi.id],
            ),
            RecordOperand.literal(
                "region_offset_bytes",
                abi.region_offset_bytes - abi_fact.region_base_bytes,
            ),
            RecordOperand.literal("size_bytes", abi.size_bytes),
            RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
            RecordOperand.literal(
                "lifetime", 2 if abi.id == output.id else 0
            ),
            RecordOperand.literal("spillable", False),
        )

    def local_copy(action: str) -> tuple[str, tuple[RecordOperand, ...]]:
        token = runtime(
            RuntimeSymbolKind.DTE_TOKEN,
            "local_token",
            action,
            (target,),
            action,
            action,
        )
        operands = (
            RecordOperand.literal("direction", 0),
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
            RecordOperand.literal("payload_bits", output.size_bytes * 8),
            RecordOperand.literal("size_bytes", output.size_bytes),
            RecordOperand.literal("hbm_address", 0),
            RecordOperand.address(
                "source_address", SemanticOperandId.SOURCE_ADDRESS,
                abs_by_abi[inputs[0].id],
            ),
            RecordOperand.address(
                "destination_address", SemanticOperandId.DESTINATION_ADDRESS,
                abs_by_abi[output.id],
            ),
        )
        return token, operands

    output_owner = (
        auxiliary
        if kind is MoeCalibrationKind.SRAM_FREE
        or (routes and output.logical_core != target)
        else owner
    )
    emit(
        output.logical_core,
        output_owner,
        RecordOpcode.SRAM_ALLOC_AT,
        alloc_operands(output),
    )
    if scratch is not None:
        emit(
            scratch.logical_core,
            auxiliary,
            RecordOpcode.SRAM_ALLOC_AT,
            alloc_operands(scratch),
        )
    if kind is MoeCalibrationKind.GROUP_GEMM:
        assert shape is not None
        m, n, k = shape
        emit(
            target,
            owner,
            RecordOpcode.SRAM_BIND,
            bind_operands(inputs, output),
        )
        emit(target, owner, RecordOpcode.MATMUL, (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, abs_by_abi[inputs[0].id]),
            RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, abs_by_abi[inputs[1].id]),
            RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, abs_by_abi[output.id]),
            RecordOperand.literal("parameters", (1, m, k, n)),
        ))
    elif kind in {MoeCalibrationKind.SWIGLU_GROUP, MoeCalibrationKind.SRAM_BIND}:
        flattened = shape[2] if shape is not None else 32
        emit(
            target,
            owner,
            RecordOpcode.SRAM_BIND,
            bind_operands((inputs[0],), output),
        )
        emit(target, owner, RecordOpcode.SWIGLU, (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, abs_by_abi[inputs[0].id]),
            RecordOperand.literal("data_address", 0),
            RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, abs_by_abi[output.id]),
            RecordOperand.literal("parameters", (flattened,)),
        ))
    elif routes:
        assert sender is not None and receiver is not None
        send_action = owner if target == sender else auxiliary
        recv_action = owner if target == receiver else auxiliary
        fsm = runtime(
            RuntimeSymbolKind.DTE_FSM,
            "flow_fsm",
            routes[0],
            (sender, receiver),
            send_action,
            recv_action,
        )
        sender_peer = runtime(
            RuntimeSymbolKind.RUNTIME_CORE,
            "sender_peer",
            str(receiver),
            (receiver,),
            None,
            None,
        )
        receiver_peer = runtime(
            RuntimeSymbolKind.RUNTIME_CORE,
            "receiver_peer",
            str(sender),
            (sender,),
            None,
            None,
        )
        token = runtime(
            RuntimeSymbolKind.DTE_TOKEN,
            "flow_token",
            recv_action,
            (receiver,),
            recv_action,
            recv_action,
        )
        emit(sender, send_action, RecordOpcode.DTE_SEND, (
            RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0),
            RecordOperand.literal("completion", 1), RecordOperand.literal("datatype", 0),
            RecordOperand.literal("reduce_op", 0),
            RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm),
            RecordOperand.literal("token", 0), RecordOperand.literal("length_bytes", output.size_bytes),
            RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, abs_by_abi[inputs[0].id]),
            RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, sender_peer),
            RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
            RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
            RecordOperand.literal("epoch", 0),
        ))
        emit(receiver, recv_action, RecordOpcode.DTE_RECV, (
            RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 0),
            RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
            RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, fsm),
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
            RecordOperand.literal("length_bytes", output.size_bytes),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, abs_by_abi[output.id]),
            RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, receiver_peer),
            RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
            RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
            RecordOperand.literal("epoch", 0),
        ))
        emit(receiver, recv_action, RecordOpcode.DTE_WAIT, (
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
        ))
    elif kind in {MoeCalibrationKind.EVENT_SET, MoeCalibrationKind.EVENT_WAIT}:
        assert peer is not None
        event_source = target if kind is MoeCalibrationKind.EVENT_SET else peer
        event_destination = peer if kind is MoeCalibrationKind.EVENT_SET else target
        set_action = owner if event_source == target else auxiliary
        wait_action = owner if event_destination == target else auxiliary
        token, operands = local_copy(owner)
        emit(target, owner, RecordOpcode.DTE_ISSUE, operands)
        emit(target, owner, RecordOpcode.DTE_WAIT, (
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
        ))
        source_core_symbol = runtime(RuntimeSymbolKind.RUNTIME_CORE, "event_source", str(event_source), (event_source,), None, None)
        destination_core_symbol = runtime(RuntimeSymbolKind.RUNTIME_CORE, "event_destination", str(event_destination), (event_destination,), None, None)
        event = runtime(RuntimeSymbolKind.EVENT_TAG, "event", recipe_id, (event_source, event_destination), set_action, wait_action)
        common = (
            RecordOperand.runtime("source_core", RuntimeOperandField.SOURCE_CORE, source_core_symbol),
            RecordOperand.runtime("destination_core", RuntimeOperandField.DESTINATION_CORE, destination_core_symbol),
            RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, event),
        )
        emit(event_source, set_action, RecordOpcode.EVENT_SET, common)
        emit(event_destination, wait_action, RecordOpcode.EVENT_WAIT, (*common, RecordOperand.literal("count", 1)))
    else:
        token, operands = local_copy(owner)
        emit(target, owner, RecordOpcode.DTE_ISSUE, operands)
        emit(target, owner, RecordOpcode.DTE_WAIT, (
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token),
        ))
    if scratch is not None:
        emit(
            scratch.logical_core,
            owner,
            RecordOpcode.SRAM_FREE,
            (
                RecordOperand.address(
                    "symbol",
                    SemanticOperandId.SYMBOL,
                    label_by_abi[scratch.id],
                ),
            ),
        )

    return _Recipe(
        recipe_id,
        target,
        target_binding.runtime_core_id,
        peer,
        tuple(sorted({core for core, _record in records}, key=lambda item: (item.die_id, item.local_core_id))),
        tuple(records),
        tuple(sorted(runtime_symbols.values(), key=lambda item: item.id)),
        tuple(sorted(runtime_definitions.values(), key=lambda item: item.symbol.id)),
        tuple(sorted(program_symbols.values(), key=lambda item: item.id)),
        tuple(sorted(program_definitions.values(), key=lambda item: item.symbol.id)),
        tuple(sorted(abis, key=lambda item: item.id)),
        tuple(sorted(item.id for item in inputs)),
        (output.id,),
        routes,
    )


def _lower_fragment(
    source: MoeSwizzleCalibrationProgramSource, recipe: _Recipe
) -> CommandFragment:
    by_core: dict[LogicalCoreRef, list[RelocatableRecord]] = defaultdict(list)
    for core, record in recipe.records:
        by_core[core].append(record)
    runtime_index = {item.id: item for item in recipe.runtime_symbols}
    program_index = {item.id: item for item in recipe.program_symbols}
    abi_by_binding = {item.binding_id: item for item in recipe.buffer_abi}
    abi_by_storage = {item.storage_id: item for item in recipe.buffer_abi}
    streams = []
    for core in sorted(by_core, key=lambda item: (item.die_id, item.local_core_id)):
        runtime_relocations = []
        address_relocations = []
        for record_index, record in enumerate(by_core[core]):
            for operand in record.operands:
                if operand.runtime_field is not None:
                    runtime_relocations.append(RuntimeRelocation(record_index, operand.runtime_field, operand.symbol_ref))
                elif operand.operand_id is not None:
                    symbol = program_index[operand.symbol_ref]
                    address_relocations.append(AddressRelocation(record_index, operand.operand_id, symbol.kind, symbol.id, 0))
        streams.append(CoreFragmentStream(
            core,
            tuple(by_core[core]),
            tuple(sorted(runtime_relocations, key=lambda item: (item.record_index, tuple(RuntimeOperandField).index(item.field)))),
            tuple(sorted(address_relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ))
    result = CommandFragment.create(
        producer_pass=_LOWERING,
        source_global_dag_id=source.id,
        kind=FragmentKind.MOE_SWIZZLE_CALIBRATION,
        claimed_action_ids=tuple(sorted({record.source_global_action_id for _core, record in recipe.records})),
        core_streams=tuple(streams),
        runtime_symbols=tuple(runtime_index.values()),
        program_symbols=tuple(program_index.values()),
        buffer_abi=recipe.buffer_abi,
        state_abi=(),
    )
    result.validate()
    return result


def _link_manifest(
    whole: MoeSwizzleStandardLinkedProgram,
    source: MoeSwizzleCalibrationProgramSource,
    recipe: _Recipe,
    fragment: CommandFragment,
) -> LinkedProgramManifest:
    runtime_defs = recipe.runtime_definitions
    program_defs = recipe.program_definitions
    program_index = {item.symbol.id: item for item in program_defs}
    abi_by_binding = {item.binding_id: item for item in fragment.buffer_abi}
    abi_by_storage = {item.storage_id: item for item in fragment.buffer_abi}
    address_bindings = []
    for stream in fragment.core_streams:
        for relocation in stream.address_relocations:
            symbol = program_index[relocation.symbol_ref].symbol
            if symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS:
                abi = abi_by_binding[symbol.source_ref]
            elif symbol.kind is ProgramSymbolKind.SRAM_LABEL:
                abi = abi_by_storage[symbol.source_ref]
            else:
                record = stream.records[relocation.record_index]
                label_ref = next(item.symbol_ref for item in record.operands if item.operand_id is SemanticOperandId.LABEL_SYMBOL)
                abi = abi_by_storage[program_index[label_ref].symbol.source_ref]
            address_bindings.append(AddressOperandBinding(
                fragment.id,
                stream.logical_core,
                relocation.record_index,
                relocation.operand_id,
                (abi.id,),
                (abi.tensor_slice,),
            ))
    event_symbols = {
        record.operands[2].symbol_ref
        for stream in fragment.core_streams
        for record in stream.records
        if record.opcode in (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT)
    }
    interface = FragmentInterface(
        fragment.id,
        (),
        tuple(sorted(item.id for item in fragment.runtime_symbols)),
        (),
        tuple(sorted(item.id for item in fragment.program_symbols)),
        tuple(EventCredit(item, 1) for item in sorted(event_symbols)),
        tuple(EventCredit(item, 1) for item in sorted(event_symbols)),
    )
    core_bindings = tuple(
        _source_core_binding(whole, core) for core in recipe.cores
    )
    linked_streams = tuple(
        LinkedCoreStream(
            stream.logical_core,
            _source_core_binding(whole, stream.logical_core).runtime_core_id,
            tuple(
                LinkedRecordRef(fragment.id, index, record.source_global_action_id)
                for index, record in enumerate(stream.records)
            ),
        )
        for stream in fragment.core_streams
    )
    start_defs = []
    starts = []
    for stream in fragment.core_streams:
        first = stream.records[0].source_global_action_id
        symbol = RuntimeSymbol(
            _id("moe_swizzle_calibration_start", (source.id, stream.logical_core, first)),
            RuntimeSymbolKind.START_TAG,
            first,
        )
        start_defs.append(RuntimeSymbolDefinition(symbol, (stream.logical_core,), None, None))
        starts.append(LogicalStartEvent(stream.logical_core, symbol.id, 1))
    inputs = tuple(sorted((
        ManifestInputDigest(
            ManifestInputKind.MOE_SWIZZLE_CALIBRATION_SOURCE,
            source.id,
            source.schema_version,
            canonical_digest(source),
        ),
        ManifestInputDigest(
            ManifestInputKind.COMMAND_FRAGMENT,
            fragment.id,
            fragment.schema_version,
            canonical_digest(fragment),
        ),
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    active = recipe.cores
    result = LinkedProgramManifest.create(
        producer_pass=_LINKER,
        capabilities=0,
        source_ir1_id=source.source_ir1_id,
        source_projection_id=source.id,
        source_schedule_set_id=source.id,
        source_global_dag_id=source.id,
        input_digests=inputs,
        fragments=(fragment,),
        fragment_interfaces=(interface,),
        core_bindings=core_bindings,
        core_streams=linked_streams,
        runtime_symbol_definitions=tuple(sorted((*runtime_defs, *start_defs), key=lambda item: item.symbol.id)),
        program_symbol_definitions=program_defs,
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id, item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=(),
        core_groups=(),
        envelope=ProgramControlEnvelope(
            active,
            tuple(starts),
            (source.target_logical_core,),
            active,
            (source.target_logical_core,),
            EmptyCoreAckPolicy.INCLUDE_EMPTY,
            ProgramFailurePolicy.ABORT_ALL,
        ),
    )
    result.validate()
    return result


def build_moe_swizzle_calibration_standard_linked_program(
    source: MoeSwizzleStandardLinkedProgram,
    kind: MoeCalibrationKind,
    shape: tuple[int, int, int] | None,
    program_artifact_sha256: str | None = None,
) -> MoeSwizzleCalibrationStandardLinkedProgram:
    """Build one real isolated standard artifact from frozen whole-program truth."""

    _validate_source_once(source)
    recipe = _build_recipe(source, kind, shape)
    expected = expected_moe_swizzle_calibration_record_quotient(kind)
    fragment_preview = _lower_fragment(
        MoeSwizzleCalibrationProgramSource.create(
            source_ir1_id=source.ir1.id,
            source_linked_program_id=source.id,
            source_linked_program_digest=canonical_digest(source),
            source_fragment_id=source.fragment.id,
            kind=kind,
            shape=shape,
            target_logical_core=recipe.target_core,
            target_runtime_core_id=recipe.target_runtime_core_id,
            target_records=(),
            record_quotient=tuple(MoeSwizzleCalibrationRecordCount(opcode, count) for opcode, count in expected),
            auxiliary_opcodes=tuple(opcode for opcode, _count in expected if opcode is not required_moe_swizzle_calibration_target_opcode(kind)),
            source_route_resource_refs=recipe.route_resource_refs,
            input_buffer_abi_ids=recipe.input_abi_ids,
            output_buffer_abi_ids=recipe.output_abi_ids,
        ) if kind is MoeCalibrationKind.TERMINAL_DONE else _source_with_target_placeholder(source, kind, shape, recipe, expected),
        recipe,
    )
    target_opcode = required_moe_swizzle_calibration_target_opcode(kind)
    target_records = ()
    if target_opcode is not None:
        target_stream = next(item for item in fragment_preview.core_streams if item.logical_core == recipe.target_core)
        target_indices = tuple(index for index, record in enumerate(target_stream.records) if record.opcode is target_opcode)
        if len(target_indices) != 1:
            raise SchemaError("target opcode is not unique on its target core", path="moe_swizzle_calibration.fragment")
        target_records = (MoeSwizzleCalibrationTargetRecord(recipe.target_core, target_indices[0], target_opcode),)
    calibration_source = MoeSwizzleCalibrationProgramSource.create(
        source_ir1_id=source.ir1.id,
        source_linked_program_id=source.id,
        source_linked_program_digest=canonical_digest(source),
        source_fragment_id=source.fragment.id,
        kind=kind,
        shape=shape,
        target_logical_core=recipe.target_core,
        target_runtime_core_id=recipe.target_runtime_core_id,
        target_records=target_records,
        record_quotient=tuple(MoeSwizzleCalibrationRecordCount(opcode, count) for opcode, count in expected),
        auxiliary_opcodes=tuple(opcode for opcode, _count in expected if opcode is not target_opcode),
        source_route_resource_refs=recipe.route_resource_refs,
        input_buffer_abi_ids=recipe.input_abi_ids,
        output_buffer_abi_ids=recipe.output_abi_ids,
    )
    fragment = _lower_fragment(calibration_source, recipe)
    manifest = _link_manifest(source, calibration_source, recipe, fragment)
    result = MoeSwizzleCalibrationStandardLinkedProgram.create(
        source=calibration_source,
        fragment=fragment,
        manifest=manifest,
        program_io=None,
    )
    if program_artifact_sha256 is None:
        return result
    from ..passes.build_moe_swizzle_calibration_program_io import (
        build_moe_swizzle_calibration_program_io,
    )
    program_io = build_moe_swizzle_calibration_program_io(
        result, program_artifact_sha256
    )
    return MoeSwizzleCalibrationStandardLinkedProgram.create(
        source=calibration_source,
        fragment=fragment,
        manifest=manifest,
        program_io=program_io,
    )


def _source_with_target_placeholder(
    source: MoeSwizzleStandardLinkedProgram,
    kind: MoeCalibrationKind,
    shape: tuple[int, int, int] | None,
    recipe: _Recipe,
    expected: tuple[tuple[RecordOpcode, int], ...],
) -> MoeSwizzleCalibrationProgramSource:
    """Create the sole temporary source needed to discover a local record index."""

    target = required_moe_swizzle_calibration_target_opcode(kind)
    assert target is not None
    # Record indices are recipe-local and independent of source id.  The
    # placeholder is replaced immediately and never escapes this builder.
    by_core = [record for core, record in recipe.records if core == recipe.target_core]
    indices = tuple(index for index, record in enumerate(by_core) if record.opcode is target)
    if len(indices) != 1:
        raise SchemaError("target opcode is not unique on its target core", path="moe_swizzle_calibration.recipe")
    return MoeSwizzleCalibrationProgramSource.create(
        source_ir1_id=source.ir1.id,
        source_linked_program_id=source.id,
        source_linked_program_digest=canonical_digest(source),
        source_fragment_id=source.fragment.id,
        kind=kind,
        shape=shape,
        target_logical_core=recipe.target_core,
        target_runtime_core_id=recipe.target_runtime_core_id,
        target_records=(MoeSwizzleCalibrationTargetRecord(recipe.target_core, indices[0], target),),
        record_quotient=tuple(MoeSwizzleCalibrationRecordCount(opcode, count) for opcode, count in expected),
        auxiliary_opcodes=tuple(opcode for opcode, _count in expected if opcode is not target),
        source_route_resource_refs=recipe.route_resource_refs,
        input_buffer_abi_ids=recipe.input_abi_ids,
        output_buffer_abi_ids=recipe.output_abi_ids,
    )


__all__ = ["build_moe_swizzle_calibration_standard_linked_program"]
