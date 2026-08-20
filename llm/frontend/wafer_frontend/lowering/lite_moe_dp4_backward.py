"""Exact leaf lowering for the isolated S3-Lite MoE WGRAD-only overlay."""

from __future__ import annotations

import struct

from ..errors import SchemaError
from ..passes.lite_moe_dp4_backward import validate_lite_moe_dp4_backward
from ..passes.lite_moe_dp4_n6 import validate_lite_moe_dp4_infer_n6_intent
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
    RuntimeSymbolKind,
    SemanticOperandId,
    StateABI,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.lite_moe import LiteMoeStaticTrace
from ..schema.lite_moe_dp4_backward import LiteMoeDp4Backward
from ..schema.lite_moe_dp4_execution import (
    LiteMoeDp4GlobalDag,
    LiteMoeDp4Projection,
    LiteMoeDp4Scheduled,
)
from ..schema.lite_moe_dp4 import LiteMoeDp4N4IR1
from ..schema.lite_moe_dp4_n6 import LiteMoeDp4InferN6Intent
from .lite_moe import (
    _absolute_symbol,
    _dte_record,
    _hbm_symbol,
    _label_symbol,
    _region_symbol,
    _runtime_symbol,
    _unique_by_id,
)


_PRODUCER = "lite_moe_dp4_backward_lowering"
_RUNTIME_ORDER = {field: index for index, field in enumerate(RuntimeOperandField)}


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _buffer(
    *,
    schedule_id: str,
    logical_core: LogicalCoreRef,
    region_ref: str,
    offset: int,
    size: int,
    value_id: str,
    shape: tuple[int, ...],
    dtype: DType,
    layout: str,
    ownership: BufferOwnership = BufferOwnership.OWNED,
    alias_of: str | None = None,
    storage_id: str | None = None,
) -> BufferABI:
    binding_semantic = {
        "schedule_id": schedule_id,
        "logical_core": logical_core,
        "value_id": value_id,
        "region_ref": region_ref,
        "offset": offset,
        "size": size,
        "layout": layout,
        "alias_of": alias_of,
    }
    binding_id = stable_artifact_id(
        "s3_lite_moe_backward_binding",
        binding_semantic,
        schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
    )
    semantic = {
        "schedule_id": schedule_id,
        "binding_id": binding_id,
        "value_id": value_id,
        "logical_core": logical_core,
        "tensor_slice": TensorSlice(value_id, tuple(0 for _ in shape), shape),
        "region_ref": region_ref,
        "region_offset_bytes": offset,
        "size_bytes": size,
        "alignment_bytes": 64,
        "banks": (),
        "storage_id": storage_id or f"s3_lite.moe.backward.storage.{binding_id}",
        "alias_of": alias_of,
        "lifetime_start": 1,
        "lifetime_end_exclusive": 100,
        "dtype": dtype,
        "layout": layout,
        "ownership": ownership,
    }
    result = BufferABI(
        stable_artifact_id(
            "buffer_abi",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        **semantic,
    )
    result.validate("buffer")
    return result


def _alias(
    root: BufferABI,
    *,
    value_id: str,
    size: int,
    shape: tuple[int, ...],
    offset: int,
    layout: str | None = None,
) -> BufferABI:
    return _buffer(
        schedule_id=root.schedule_id,
        logical_core=root.logical_core,
        region_ref=root.region_ref,
        offset=root.region_offset_bytes + offset,
        size=size,
        value_id=value_id,
        shape=shape,
        dtype=root.dtype,
        layout=layout or f"{root.layout}_view",
        ownership=BufferOwnership.ALIASED,
        alias_of=root.binding_id,
        storage_id=root.storage_id,
    )


def _runtime_cores(
    source: LiteMoeDp4N4IR1,
    intent: LiteMoeDp4InferN6Intent,
) -> dict[int, tuple[LogicalCoreRef, int]]:
    result: dict[int, tuple[LogicalCoreRef, int]] = {}
    for die_id in range(4):
        used = {
            abi.logical_core
            for abi in intent.buffer_abis
            if abi.logical_core.die_id == die_id
        }
        if len(used) != 1:
            raise SchemaError(
                "backward preview requires one scheduled execution core per die",
                path="intent.buffer_abis",
            )
        logical = next(iter(used))
        die = next(item for item in source.graph.fabric.dies if item.id == die_id)
        core = next(
            item for item in die.cores
            if item.local_core_id == logical.local_core_id
        )
        result[die_id] = (logical, core.runtime_core_id)
    return result


def _fragment(
    overlay: LiteMoeDp4Backward,
    *,
    kind: FragmentKind,
    claims: tuple[str, ...],
    core: LogicalCoreRef,
    records: tuple[RelocatableRecord, ...],
    runtime_symbols: tuple[object, ...] = (),
    program_symbols: tuple[object, ...] = (),
    runtime_relocations: tuple[RuntimeRelocation, ...] = (),
    address_relocations: tuple[AddressRelocation, ...] = (),
    buffer_abi: tuple[BufferABI, ...] = (),
    state_abi: tuple[StateABI, ...] = (),
) -> CommandFragment:
    result = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=overlay.id,
        kind=kind,
        claimed_action_ids=tuple(sorted(claims)),
        core_streams=(CoreFragmentStream(
            core,
            records,
            tuple(sorted(runtime_relocations, key=lambda item: (
                item.record_index, _RUNTIME_ORDER[item.field],
            ))),
            tuple(sorted(address_relocations, key=lambda item: (
                item.record_index, int(item.operand_id),
            ))),
        ),),
        runtime_symbols=_unique_by_id(runtime_symbols),
        program_symbols=_unique_by_id(program_symbols),
        buffer_abi=tuple(sorted(
            {item.id: item for item in buffer_abi}.values(),
            key=lambda item: item.id,
        )),
        state_abi=tuple(sorted(state_abi, key=lambda item: item.id)),
    )
    result.validate("lite_moe_backward_fragment")
    return result


def _alloc(action: str, abi: BufferABI, runtime_core: int):
    region = _region_symbol(abi.region_ref)
    label = _label_symbol(abi, runtime_core)
    record = RelocatableRecord(action, RecordOpcode.SRAM_ALLOC_AT, (
        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.id),
        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
        RecordOperand.literal("region_offset_bytes", abi.region_offset_bytes),
        RecordOperand.literal("size_bytes", abi.size_bytes),
        RecordOperand.literal("alignment_bytes", abi.alignment_bytes),
        RecordOperand.literal("lifetime", 0),
        RecordOperand.literal("spillable", False),
    ))
    return record, (region, label), (
        AddressRelocation(0, SemanticOperandId.REGION_NAME,
                          ProgramSymbolKind.SRAM_REGION, region.id, 0),
        AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL,
                          ProgramSymbolKind.SRAM_LABEL, label.id, 0),
    )


def _free(action: str, abi: BufferABI, runtime_core: int):
    label = _label_symbol(abi, runtime_core)
    return (
        RelocatableRecord(action, RecordOpcode.SRAM_FREE, (
            RecordOperand.address("symbol", SemanticOperandId.SYMBOL, label.id),
        )),
        label,
    )


def _bind(action: str, inputs: tuple[BufferABI, ...], output: BufferABI,
          runtime_core: int) -> tuple[RelocatableRecord, tuple[object, ...],
                                      tuple[AddressRelocation, ...]]:
    labels = tuple(_label_symbol(item, runtime_core) for item in inputs)
    output_label = _label_symbol(output, runtime_core)
    record = RelocatableRecord(action, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", len(inputs)),
        *(
            RecordOperand.address(
                f"input_label_{index}",
                SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index),
                labels[index].id,
            )
            if index < len(labels)
            else RecordOperand.literal(f"input_label_{index}", 0)
            for index in range(16)
        ),
        RecordOperand.address(
            "output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
            output_label.id,
        ),
    ))
    relocations = tuple(
        AddressRelocation(
            0,
            SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index),
            ProgramSymbolKind.SRAM_LABEL,
            label.id,
            0,
        )
        for index, label in enumerate(labels)
    ) + (AddressRelocation(
        0, SemanticOperandId.SRAM_BIND_OUTPUT,
        ProgramSymbolKind.SRAM_LABEL, output_label.id, 0,
    ),)
    return record, (*labels, output_label), relocations


def _state_abi(item, source: LiteMoeDp4N4IR1) -> StateABI:
    home = next(
        space for space in source.graph.persistent_state_manifest.address_spaces
        if space.die_id == item.home_die_id
    )
    declaration, binding = item.declaration, item.binding
    return StateABI.create(
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


def _local_reduce_source_symbol(
    abi: BufferABI,
    action_id: str,
) -> ProgramSymbol:
    semantic = {
        "schedule_id": abi.schedule_id,
        "binding_id": abi.binding_id,
        "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS),
        "purpose": "local_reduce_rank_major_source",
        "action_id": action_id,
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


def lower_lite_moe_dp4_backward(
    overlay: LiteMoeDp4Backward,
    n4: LiteMoeDp4N4IR1,
    projection: LiteMoeDp4Projection,
    schedule: LiteMoeDp4Scheduled,
    global_dag: LiteMoeDp4GlobalDag,
    n6_intent: LiteMoeDp4InferN6Intent,
    trace: LiteMoeStaticTrace,
) -> tuple[CommandFragment, ...]:
    """Lower the exact typed DP4 WGRAD-only overlay into 32 canonical leaves."""

    validate_lite_moe_dp4_backward(overlay, overlay.train_forward)
    formal = overlay.train_forward.forward
    if (
        n4 != formal.n4
        or projection != formal.projection
        or schedule != formal.schedule
        or global_dag != formal.global_dag
        or trace != formal.adapter.spec.trace
    ):
        raise SchemaError(
            "backward formal sources do not match embedded training-forward",
            path="lite_moe_dp4_backward",
        )
    validate_lite_moe_dp4_infer_n6_intent(n6_intent, formal)
    cores = _runtime_cores(n4, n6_intent)
    existing = {item.id: item for item in n6_intent.buffer_abis}
    high = {
        die: _align(max(
            item.region_offset_bytes + item.size_bytes
            for item in n6_intent.buffer_abis
            if item.logical_core.die_id == die
        ))
        for die in range(4)
    }
    regions = {
        die: next(
            item.region_ref for item in n6_intent.buffer_abis
            if item.logical_core.die_id == die
        )
        for die in range(4)
    }

    def reserve(die: int, size: int, value: str, shape: tuple[int, ...],
                dtype: DType, layout: str, *,
                ownership: BufferOwnership = BufferOwnership.OWNED) -> BufferABI:
        offset = high[die]
        result = _buffer(
            schedule_id=overlay.id,
            logical_core=cores[die][0],
            region_ref=regions[die],
            offset=offset,
            size=size,
            value_id=value,
            shape=shape,
            dtype=dtype,
            layout=layout,
            ownership=ownership,
        )
        high[die] = _align(offset + size)
        return result

    down_units = {
        item.node_ref: item
        for item in n6_intent.compute_units
        if item.node_ref.endswith(".down")
    }
    saved: dict[int, BufferABI] = {}
    for unit in overlay.token_wgrads:
        source_unit = down_units[unit.down_node_ref]
        source_abi = existing[source_unit.inputs[0].buffer_abi_ref]
        saved[unit.token_index] = reserve(
            unit.home_die_id, 64, unit.tape_value_ref, (1, 32),
            DType.FP16, "s3_lite_moe_saved_activation",
            ownership=BufferOwnership.BORROWED,
        )
        if source_abi.size_bytes != 64:
            raise SchemaError("forward saved activation span changed", path="n6_intent")

    upstream_source: dict[int, BufferABI] = {}
    upstream_home: dict[int, BufferABI] = {}
    remote = {item.token_index: item for item in overlay.remote_gradients}
    for unit in overlay.token_wgrads:
        transport = remote.get(unit.token_index)
        source_die = transport.source_die_id if transport is not None else unit.home_die_id
        source_ref = (
            transport.source_gradient_ref
            if transport is not None
            else unit.upstream_gradient_ref
        )
        source = reserve(
            source_die, 32, source_ref, (1, 16),
            DType.FP16, "s3_lite_moe_upstream_gradient",
            ownership=BufferOwnership.BORROWED,
        )
        upstream_source[unit.token_index] = source
        upstream_home[unit.token_index] = (
            reserve(
                unit.home_die_id, 32, transport.received_gradient_ref,
                (1, 16), DType.FP16, "s3_lite_moe_received_gradient",
            )
            if source_die != unit.home_die_id else source
        )

    roots: dict[int, BufferABI] = {}
    contributions: dict[str, BufferABI] = {}
    for reduce in overlay.expert_reduces:
        root = reserve(
            reduce.home_die_id, 4096, reduce.root_buffer_ref, (1024,),
            DType.FP32, "s3_lite_moe_wgrad_root",
        )
        roots[reduce.expert_index] = root
        for ref, offset in zip(reduce.contribution_refs, reduce.input_offsets, strict=True):
            contributions[ref] = _alias(
                root, value_id=ref, size=2048, shape=(512,), offset=offset
            )

    weights: dict[int, BufferABI] = {}
    updated: dict[int, BufferABI] = {}
    for item in overlay.trainable_down_states:
        weight = reserve(
            item.home_die_id, 1024, item.declaration.id, (32, 16),
            DType.FP16, "s3_lite_moe_trainable_down_weight",
        )
        weights[item.expert_index] = weight
        updated[item.expert_index] = _alias(
            weight, value_id=f"{item.declaration.id}.updated",
            size=1024, shape=(32, 16), offset=0, layout=weight.layout,
        )

    fragments: list[CommandFragment] = []
    # One blocking trainable-state load per expert.
    for item in overlay.trainable_down_states:
        action = f"{overlay.id}.expert{item.expert_index}.weight_load"
        weight = weights[item.expert_index]
        alloc, alloc_symbols, alloc_relocs = _alloc(action, weight, cores[item.home_die_id][1])
        hbm, local = _hbm_symbol(item.binding.id), _absolute_symbol(weight)
        load = RelocatableRecord(action, RecordOpcode.LSU_LOAD, (
            RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
            RecordOperand.literal("size_bytes", 1024),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, local.id),
        ))
        fragments.append(_fragment(
            overlay, kind=FragmentKind.STATE_IO, claims=(action,),
            core=cores[item.home_die_id][0], records=(alloc, load),
            program_symbols=(*alloc_symbols, hbm, local),
            address_relocations=(
                *alloc_relocs,
                AddressRelocation(1, SemanticOperandId.HBM_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm.id, 0),
                AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, local.id, 0),
            ),
            buffer_abi=(weight,), state_abi=(_state_abi(item, n4),),
        ))

    # Six exact 32-byte remote-gradient endpoint pairs.
    for unit in overlay.remote_gradients:
        send_id, recv_id, wait_id = unit.send_ref, unit.recv_ref, unit.wait_ref
        source = upstream_source[unit.token_index]
        destination = upstream_home[unit.token_index]
        fsm = _runtime_symbol(RuntimeSymbolKind.DTE_FSM, unit.forward_combine_flow_ref,
                              ("lite_moe_backward", unit.id, "fsm"))
        token = _runtime_symbol(RuntimeSymbolKind.DTE_TOKEN, unit.id,
                                ("lite_moe_backward", unit.id, "token"))
        send_peer = _runtime_symbol(RuntimeSymbolKind.RUNTIME_CORE, unit.reverse_pair_route_ref,
                                    ("lite_moe_backward", unit.id, "send_peer"))
        recv_peer = _runtime_symbol(RuntimeSymbolKind.RUNTIME_CORE, unit.reverse_pair_route_ref,
                                    ("lite_moe_backward", unit.id, "recv_peer"))
        source_symbol, destination_symbol = _absolute_symbol(source), _absolute_symbol(destination)
        send = _dte_record(
            action_id=send_id, is_send=True, length_bytes=32,
            address=source_symbol, fsm=fsm, peer=send_peer, token=None,
        )
        recv = _dte_record(
            action_id=recv_id, is_send=False, length_bytes=32,
            address=destination_symbol, fsm=fsm, peer=recv_peer, token=token,
        )
        wait = RelocatableRecord(
            wait_id, RecordOpcode.DTE_WAIT,
            (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
        )
        salloc, ssymbols, srelocs = _alloc(
            send_id, source, cores[unit.source_die_id][1]
        )
        sfree, sfree_symbol = _free(
            send_id, source, cores[unit.source_die_id][1]
        )
        fragments.append(_fragment(
            overlay, kind=FragmentKind.MOE_TRANSFER, claims=(send_id,),
            core=cores[unit.source_die_id][0], records=(salloc, send, sfree),
            runtime_symbols=(fsm, send_peer),
            program_symbols=(*ssymbols, source_symbol, sfree_symbol),
            runtime_relocations=(
                RuntimeRelocation(1, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(1, RuntimeOperandField.PEER_CORE, send_peer.id),
            ),
            address_relocations=(
                *srelocs,
                AddressRelocation(1, SemanticOperandId.SOURCE_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
                AddressRelocation(2, SemanticOperandId.SYMBOL,
                                  ProgramSymbolKind.SRAM_LABEL, sfree_symbol.id, 0),
            ),
            buffer_abi=(source,),
        ))
        dalloc, dsymbols, drelocs = _alloc(recv_id, destination, cores[unit.destination_die_id][1])
        fragments.append(_fragment(
            overlay, kind=FragmentKind.MOE_TRANSFER,
            claims=(recv_id, wait_id), core=cores[unit.destination_die_id][0],
            records=(dalloc, recv, wait),
            runtime_symbols=(fsm, token, recv_peer),
            program_symbols=(*dsymbols, destination_symbol),
            runtime_relocations=(
                RuntimeRelocation(1, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
                RuntimeRelocation(1, RuntimeOperandField.PEER_CORE, recv_peer.id),
                RuntimeRelocation(2, RuntimeOperandField.DTE_TOKEN, token.id),
            ),
            address_relocations=(
                *drelocs,
                AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0),
            ),
            buffer_abi=(destination,),
        ))

    # Eight token outer-product WGRAD records.
    for unit in overlay.token_wgrads:
        action = unit.id
        activation, gradient = saved[unit.token_index], upstream_home[unit.token_index]
        output = contributions[unit.contribution_ref]
        runtime_core = cores[unit.home_die_id][1]
        records: list[RelocatableRecord] = []
        symbols: list[object] = []
        relocs: list[AddressRelocation] = []
        for abi in (
            activation,
            *((roots[unit.expert_index],) if unit.offset_bytes == 0 else ()),
        ):
            alloc, alloc_symbols, alloc_relocs = _alloc(action, abi, runtime_core)
            base = len(records); records.append(alloc); symbols.extend(alloc_symbols)
            relocs.extend(AddressRelocation(
                base, item.operand_id, item.symbol_kind, item.symbol_ref, item.addend
            ) for item in alloc_relocs)
        if unit.token_index not in remote:
            alloc, alloc_symbols, alloc_relocs = _alloc(action, gradient, runtime_core)
            base = len(records); records.append(alloc); symbols.extend(alloc_symbols)
            relocs.extend(AddressRelocation(
                base, item.operand_id, item.symbol_kind, item.symbol_ref, item.addend
            ) for item in alloc_relocs)
        bind, bind_symbols, bind_relocs = _bind(action, (activation, gradient), output, runtime_core)
        bind_index = len(records); records.append(bind); symbols.extend(bind_symbols)
        relocs.extend(AddressRelocation(
            bind_index, item.operand_id, item.symbol_kind, item.symbol_ref, item.addend
        ) for item in bind_relocs)
        a, g, out = _absolute_symbol(activation), _absolute_symbol(gradient), _absolute_symbol(output)
        matmul_index = len(records)
        records.append(RelocatableRecord(action, RecordOpcode.MATMUL, (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, a.id),
            RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, g.id),
            RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, out.id),
            RecordOperand.literal("parameters", (1, 32, 1, 16)),
        )))
        symbols.extend((a, g, out))
        relocs.extend((
            AddressRelocation(matmul_index, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                              ProgramSymbolKind.ABSOLUTE_ADDRESS, a.id, 0),
            AddressRelocation(matmul_index, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                              ProgramSymbolKind.ABSOLUTE_ADDRESS, g.id, 0),
            AddressRelocation(matmul_index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                              ProgramSymbolKind.ABSOLUTE_ADDRESS, out.id, 0),
        ))
        for abi in (activation, gradient):
            free, label = _free(action, abi, runtime_core)
            free_index = len(records); records.append(free); symbols.append(label)
            relocs.append(AddressRelocation(
                free_index, SemanticOperandId.SYMBOL,
                ProgramSymbolKind.SRAM_LABEL, label.id, 0,
            ))
        fragments.append(_fragment(
            overlay, kind=FragmentKind.COARSE, claims=(action,),
            core=cores[unit.home_die_id][0], records=tuple(records),
            program_symbols=tuple(symbols), address_relocations=tuple(relocs),
            buffer_abi=(activation, gradient, roots[unit.expert_index], output),
        ))

    # Four exact FP32 rank-major reductions.
    for unit in overlay.expert_reduces:
        inputs = tuple(contributions[item] for item in unit.contribution_refs)
        output = contributions[unit.output_alias_ref]
        source_symbol = _local_reduce_source_symbol(inputs[0], unit.id)
        destination_symbol = _absolute_symbol(output)
        record = RelocatableRecord(unit.id, RecordOpcode.LOCAL_REDUCE, (
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
        ))
        fragments.append(_fragment(
            overlay, kind=FragmentKind.COARSE, claims=(unit.id,),
            core=cores[unit.home_die_id][0], records=(record,),
            program_symbols=(source_symbol, destination_symbol),
            address_relocations=(
                AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, source_symbol.id, 0),
                AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_symbol.id, 0),
            ),
            buffer_abi=(*inputs, output),
        ))

    # Four in-place SGD updates followed by blocking HBM stores and cleanup.
    state_by_expert = {item.expert_index: item for item in overlay.trainable_down_states}
    for unit in overlay.sgd_stores:
        action = unit.id
        weight, gradient, output = (
            weights[unit.expert_index],
            contributions[unit.gradient_alias_ref],
            updated[unit.expert_index],
        )
        runtime_core = cores[unit.home_die_id][1]
        bind, bind_symbols, bind_relocs = _bind(action, (weight, gradient), output, runtime_core)
        w, g = _absolute_symbol(weight), _absolute_symbol(gradient)
        sgd = RelocatableRecord(action, RecordOpcode.SGD_UPDATE, (
            RecordOperand.literal("weight_datatype", 1),
            RecordOperand.literal("gradient_datatype", 3),
            RecordOperand.literal("output_datatype", 1),
            RecordOperand.literal("rounding", 0),
            RecordOperand.address("weight_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, w.id),
            RecordOperand.address("gradient_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, g.id),
            RecordOperand.address("updated_weight_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, w.id),
            RecordOperand.literal("element_count", 512),
            RecordOperand.literal(
                "learning_rate_f64_bits",
                struct.unpack("<Q", struct.pack("<d", unit.learning_rate))[0],
            ),
            RecordOperand.literal("momentum_f64_bits", 0),
        ))
        state = state_by_expert[unit.expert_index]
        hbm = _hbm_symbol(state.binding.id)
        store = RelocatableRecord(action, RecordOpcode.LSU_STORE, (
            RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
            RecordOperand.literal("size_bytes", 1024),
            RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, w.id),
        ))
        free_weight, weight_label = _free(action, weight, runtime_core)
        free_root, root_label = _free(action, roots[unit.expert_index], runtime_core)
        fragments.append(_fragment(
            overlay, kind=FragmentKind.STATE_IO, claims=(action,),
            core=cores[unit.home_die_id][0],
            records=(bind, sgd, store, free_weight, free_root),
            program_symbols=(*bind_symbols, w, g, hbm, weight_label, root_label),
            address_relocations=(
                *bind_relocs,
                AddressRelocation(1, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, w.id, 0),
                AddressRelocation(1, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, g.id, 0),
                AddressRelocation(1, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, w.id, 0),
                AddressRelocation(2, SemanticOperandId.SOURCE_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, w.id, 0),
                AddressRelocation(2, SemanticOperandId.HBM_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm.id, 0),
                AddressRelocation(3, SemanticOperandId.SYMBOL,
                                  ProgramSymbolKind.SRAM_LABEL, weight_label.id, 0),
                AddressRelocation(4, SemanticOperandId.SYMBOL,
                                  ProgramSymbolKind.SRAM_LABEL, root_label.id, 0),
            ),
            buffer_abi=(weight, gradient, output, roots[unit.expert_index]),
            state_abi=(_state_abi(state, n4),),
        ))

    result = tuple(sorted(fragments, key=lambda item: item.id))
    if len(result) != 32:
        raise SchemaError("backward lowering must emit exactly 32 leaves", path="fragments")
    return result


def validate_lite_moe_dp4_backward_fragments(
    fragments: tuple[CommandFragment, ...],
    overlay: LiteMoeDp4Backward,
    n4: LiteMoeDp4N4IR1,
    projection: LiteMoeDp4Projection,
    schedule: LiteMoeDp4Scheduled,
    global_dag: LiteMoeDp4GlobalDag,
    n6_intent: LiteMoeDp4InferN6Intent,
    trace: LiteMoeStaticTrace,
) -> None:
    """Rebuild and compare the complete typed backward leaf quotient."""

    if type(fragments) is not tuple:
        raise SchemaError("backward fragments must be a tuple", path="fragments")
    for index, fragment in enumerate(fragments):
        if type(fragment) is not CommandFragment:
            raise SchemaError(
                "backward fragment must be a CommandFragment",
                path=f"fragments[{index}]",
            )
        fragment.validate(f"fragments[{index}]")
    expected = lower_lite_moe_dp4_backward(
        overlay, n4, projection, schedule, global_dag, n6_intent, trace
    )
    if fragments != expected:
        raise SchemaError(
            "backward fragments are not the exact typed lowering quotient",
            path="fragments",
        )


__all__ = [
    "lower_lite_moe_dp4_backward",
    "validate_lite_moe_dp4_backward_fragments",
]
