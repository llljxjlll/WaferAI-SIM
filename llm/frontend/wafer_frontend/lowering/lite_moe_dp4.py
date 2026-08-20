"""Exact public record lowering for fixed four-die S3-Lite MoE inference."""

from __future__ import annotations

from math import prod

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
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
from ..schema.ir0 import GemmWorkload, SwiGluWorkload
from ..schema.ir2 import BufferOwnership, TensorSlice
from ..schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase, LiteMoeDp4TaskKind
from ..schema.lite_moe_dp4_n6 import (
    LiteMoeDp4ComputeUnit,
    LiteMoeDp4DteUnit,
    LiteMoeDp4InferN6Intent,
)
from ..schema.lite_moe_dp4_train_forward import (
    LiteMoeDp4TapeBuffer,
    LiteMoeDp4TapeCopy,
    LiteMoeDp4TrainForward,
)
from ..schema.lite_moe_n6 import LiteMoeStateLoadUnit
from .lite_moe import (
    _absolute_symbol,
    _action_core,
    _decorate_lifecycle,
    _dte_record,
    _hbm_symbol,
    _label_symbol,
    _operand_abi,
    _runtime_symbol,
    _region_symbol,
    _unique_by_id,
)


_PRODUCER_PASS = "lite_moe_dp4_infer_lowering"
_RUNTIME_FIELD_ORDER = {field: index for index, field in enumerate(RuntimeOperandField)}


def _indices(intent: LiteMoeDp4InferN6Intent, source: LiteMoeDp4ExecutionCase):
    intent.validate("intent")
    source.validate("source")
    if (
        intent.source_case_id != source.id
        or intent.source_global_id != source.global_dag.id
        or intent.source_schedule_id != source.schedule.id
        or intent.source_projection_id != source.projection.id
        or intent.source_n4_id != source.n4.id
    ):
        raise SchemaError("DP4 lowering sources disagree", path="source")
    return (
        {item.id: item for item in intent.buffer_abis},
        {item.id: item for item in source.global_dag.actions},
        {item.id: item for die in source.projection.dies for item in die.tasks},
    )


def _fragment(
    source: LiteMoeDp4ExecutionCase,
    *,
    kind: FragmentKind,
    claims: tuple[str, ...],
    stream: CoreFragmentStream,
    runtime_symbols=(),
    program_symbols=(),
    buffer_abi=(),
    state_abi=(),
) -> CommandFragment:
    result = CommandFragment.create(
        producer_pass=_PRODUCER_PASS,
        source_global_dag_id=source.global_dag.id,
        kind=kind,
        claimed_action_ids=tuple(sorted(claims)),
        core_streams=(stream,),
        runtime_symbols=_unique_by_id(tuple(runtime_symbols)),
        program_symbols=_unique_by_id(tuple(program_symbols)),
        buffer_abi=tuple(sorted({item.id: item for item in buffer_abi}.values(), key=lambda item: item.id)),
        state_abi=tuple(sorted(state_abi, key=lambda item: item.id)),
    )
    result.validate("lite_moe_dp4_fragment")
    return _decorate_lifecycle(
        result,
        source.global_dag,
        source.schedule,
        source.n4,
    )


def lower_lite_moe_dp4_compute_unit(
    unit: LiteMoeDp4ComputeUnit,
    intent: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
) -> CommandFragment:
    if type(unit) is not LiteMoeDp4ComputeUnit or unit not in intent.compute_units:
        raise SchemaError("unit is not owned by the DP4 intent", path="unit")
    unit.validate("unit")
    abi_index, actions, tasks = _indices(intent, source)
    action = actions.get(unit.action_ref)
    if action is None or action.kind is not unit.kind:
        raise SchemaError("compute unit/action mismatch", path="unit.action_ref")
    task = tasks[action.task_ref]
    if task.node_ref != unit.node_ref or task.workload != unit.workload:
        raise SchemaError("compute unit/task mismatch", path="unit")
    logical_core, runtime_core_id = _action_core(action, source.n4)
    inputs = tuple(
        _operand_abi(item, abi_index, path=f"unit.inputs[{index}]")
        for index, item in enumerate(unit.inputs)
    )
    output = _operand_abi(unit.output, abi_index, path="unit.output")
    if any(item.logical_core != logical_core for item in (*inputs, output)):
        raise SchemaError("compute views must reside on the action core", path="unit")
    if any(item.dtype is not DType.FP16 for item in (*inputs, output)):
        raise SchemaError("DP4 compute requires FP16 BufferABI", path="unit")
    if unit.kind is LiteMoeDp4TaskKind.GEMM:
        if type(unit.workload) is not GemmWorkload:
            raise SchemaError("GEMM requires GemmWorkload", path="unit.workload")
        rank_m, rank_n, rank_k = unit.workload.rank_shape
        if tuple(item.size_bytes for item in (*unit.inputs, unit.output)) != (
            2 * rank_m * rank_k,
            2 * rank_k * rank_n,
            2 * rank_m * rank_n,
        ):
            raise SchemaError("GEMM byte spans disagree with rank shape", path="unit")
        opcode = RecordOpcode.MATMUL
        parameters = (1, rank_m, rank_k, rank_n)
        data_symbol = _absolute_symbol(inputs[1])
        data_operand = RecordOperand.address(
            "data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS, data_symbol.id
        )
    else:
        if type(unit.workload) is not SwiGluWorkload:
            raise SchemaError("SWIGLU requires SwiGluWorkload", path="unit.workload")
        if tuple(item.size_bytes for item in (*unit.inputs, unit.output)) != (
            2 * prod(unit.workload.rank_input_shape),
            2 * prod(unit.workload.rank_output_shape),
        ):
            raise SchemaError("SWIGLU byte spans disagree with rank shape", path="unit")
        opcode = RecordOpcode.SWIGLU
        parameters = (prod(unit.workload.rank_output_shape),)
        data_symbol = None
        data_operand = RecordOperand.literal("data_address", 0)
    input_label = _label_symbol(inputs[0], runtime_core_id)
    output_label = _label_symbol(output, runtime_core_id)
    input_symbol = _absolute_symbol(inputs[0])
    output_symbol = _absolute_symbol(output)
    bind = RelocatableRecord(action.id, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", 1),
        RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0, input_label.id),
        *(RecordOperand.literal(f"input_label_{index}", 0) for index in range(1, 16)),
        RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT, output_label.id),
    ))
    compute = RelocatableRecord(action.id, opcode, (
        RecordOperand.literal("datatype", 1),
        RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, input_symbol.id),
        data_operand,
        RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, output_symbol.id),
        RecordOperand.literal("parameters", parameters),
    ))
    relocations = [
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_INPUT_0, ProgramSymbolKind.SRAM_LABEL, input_label.id, 0),
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_OUTPUT, ProgramSymbolKind.SRAM_LABEL, output_label.id, 0),
        AddressRelocation(1, SemanticOperandId.COMPUTE_INPUT_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, input_symbol.id, unit.inputs[0].offset_bytes),
        AddressRelocation(1, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, output_symbol.id, unit.output.offset_bytes),
    ]
    symbols = [input_label, output_label, input_symbol, output_symbol]
    if data_symbol is not None:
        relocations.append(AddressRelocation(
            1, SemanticOperandId.COMPUTE_DATA_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS, data_symbol.id,
            unit.inputs[1].offset_bytes,
        ))
        symbols.append(data_symbol)
    return _fragment(
        source,
        kind=FragmentKind.COARSE,
        claims=(action.id,),
        stream=CoreFragmentStream(
            logical_core,
            (bind, compute),
            (),
            tuple(sorted(relocations, key=lambda item: (item.record_index, int(item.operand_id)))),
        ),
        program_symbols=tuple(symbols),
        buffer_abi=(*inputs, output),
    )


def lower_lite_moe_dp4_state_load_unit(
    unit: LiteMoeStateLoadUnit,
    intent: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
) -> CommandFragment:
    if type(unit) is not LiteMoeStateLoadUnit or unit not in intent.state_loads:
        raise SchemaError("unit is not owned by the DP4 intent", path="unit")
    unit.validate("unit")
    abi_index, actions, tasks = _indices(intent, source)
    action = actions.get(unit.action_ref)
    if action is None or action.kind is not LiteMoeDp4TaskKind.DMA_IN:
        raise SchemaError("state unit/action mismatch", path="unit.action_ref")
    task = tasks[action.task_ref]
    if task.state_ref != unit.state_ref or task.hbm_binding_ref != unit.hbm_binding_ref:
        raise SchemaError("state unit/task mismatch", path="unit")
    logical_core, _runtime_core_id = _action_core(action, source.n4)
    local = abi_index[unit.destination_buffer_abi_ref]
    if local.logical_core != logical_core or local.dtype is not DType.FP16 or local.size_bytes != unit.bytes:
        raise SchemaError("state destination geometry mismatch", path="unit")
    manifest = source.n4.graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError("state unit requires persistent manifest", path="source.n4")
    binding = next((item for item in manifest.bindings if item.id == unit.hbm_binding_ref), None)
    declaration = next((item for item in manifest.declarations if item.id == unit.state_ref), None)
    if (
        binding is None or declaration is None or binding.state_ref != declaration.id
        or binding.die_id != action.die_id or binding.size_bytes != unit.bytes
        or declaration.tensor_bytes != unit.bytes
    ):
        raise SchemaError("state HBM/declaration closure mismatch", path="unit")
    home = next((item for item in manifest.address_spaces if item.die_id == binding.die_id), None)
    if home is None:
        raise SchemaError("state binding lacks HBM space", path="unit")
    hbm = _hbm_symbol(binding.id)
    destination = _absolute_symbol(local)
    record = RelocatableRecord(action.id, RecordOpcode.LSU_LOAD, (
        RecordOperand.address("hbm_address", SemanticOperandId.HBM_ADDRESS, hbm.id),
        RecordOperand.literal("size_bytes", unit.bytes),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination.id),
    ))
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
    return _fragment(
        source,
        kind=FragmentKind.STATE_IO,
        claims=(action.id,),
        stream=CoreFragmentStream(
            logical_core,
            (record,),
            (),
            tuple(sorted((
                AddressRelocation(0, SemanticOperandId.HBM_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, hbm.id, 0),
                AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination.id, 0),
            ), key=lambda item: int(item.operand_id))),
        ),
        program_symbols=(hbm, destination),
        buffer_abi=(local,),
        state_abi=(state_abi,),
    )


def lower_lite_moe_dp4_dte_unit(
    unit: LiteMoeDp4DteUnit,
    intent: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
) -> tuple[CommandFragment, CommandFragment]:
    if type(unit) is not LiteMoeDp4DteUnit or unit not in intent.dte_units:
        raise SchemaError("unit is not owned by the DP4 intent", path="unit")
    unit.validate("unit")
    abi_index, actions, _tasks = _indices(intent, source)
    send, recv, wait = (
        actions.get(unit.send_action_ref),
        actions.get(unit.recv_action_ref),
        actions.get(unit.wait_action_ref),
    )
    if (
        send is None or recv is None or wait is None
        or (send.kind, recv.kind, wait.kind) != (
            LiteMoeDp4TaskKind.SEND, LiteMoeDp4TaskKind.RECV, LiteMoeDp4TaskKind.WAIT,
        )
        or (send.flow_ref, recv.flow_ref, wait.flow_ref) != (unit.flow_ref,) * 3
        or (send.die_id, recv.die_id, wait.die_id) != (
            unit.source_die_id, unit.destination_die_id, unit.destination_die_id,
        )
        or wait.deps != (recv.id,)
    ):
        raise SchemaError("DTE action closure mismatch", path="unit")
    binding = next((item for item in source.n4.p2p_bindings if item.id == unit.p2p_binding_ref), None)
    routes = tuple(route for group in source.n4.graph.groups for route in group.embedding.routes)
    route = next((item for item in routes if item.id == unit.pair_route_ref), None)
    if (
        binding is None or route is None
        or (binding.source_die_id, binding.destination_die_id) != (
            unit.source_die_id, unit.destination_die_id,
        )
        or (route.die_path[0], route.die_path[-1]) != (
            unit.source_die_id, unit.destination_die_id,
        )
    ):
        raise SchemaError("DTE route/binding closure mismatch", path="unit")
    source_abi = abi_index[unit.source_buffer_abi_ref]
    destination_abi = abi_index[unit.destination_buffer_abi_ref]
    source_core, _ = _action_core(send, source.n4)
    destination_core, _ = _action_core(recv, source.n4)
    if (
        source_abi.logical_core != source_core
        or destination_abi.logical_core != destination_core
        or source_abi.dtype is not DType.FP16
        or destination_abi.dtype is not DType.FP16
        or source_abi.size_bytes != unit.bytes
        or destination_abi.size_bytes != unit.bytes
    ):
        raise SchemaError("DTE endpoint BufferABI mismatch", path="unit")
    from ..schema.artifact_manifest import RuntimeSymbolKind
    fsm = _runtime_symbol(RuntimeSymbolKind.DTE_FSM, unit.channel_symbol, ("channel", unit.channel_symbol))
    token = _runtime_symbol(RuntimeSymbolKind.DTE_TOKEN, unit.token_symbol, ("recv", recv.id))
    source_peer = _runtime_symbol(RuntimeSymbolKind.RUNTIME_CORE, unit.channel_symbol, ("peer", send.id))
    destination_peer = _runtime_symbol(RuntimeSymbolKind.RUNTIME_CORE, unit.channel_symbol, ("peer", recv.id))
    source_address = _absolute_symbol(source_abi)
    destination_address = _absolute_symbol(destination_abi)
    send_record = _dte_record(
        action_id=send.id, is_send=True, length_bytes=unit.bytes,
        address=source_address, fsm=fsm, peer=source_peer, token=None,
    )
    recv_record = _dte_record(
        action_id=recv.id, is_send=False, length_bytes=unit.bytes,
        address=destination_address, fsm=fsm, peer=destination_peer, token=token,
    )
    wait_record = RelocatableRecord(
        wait.id, RecordOpcode.DTE_WAIT,
        (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
    )
    source_fragment = _fragment(
        source,
        kind=FragmentKind.MOE_TRANSFER,
        claims=(send.id,),
        stream=CoreFragmentStream(
            source_core,
            (send_record,),
            tuple(sorted((
                RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, source_peer.id),
            ), key=lambda item: (item.record_index, _RUNTIME_FIELD_ORDER[item.field]))),
            (AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source_address.id, 0),),
        ),
        runtime_symbols=(fsm, source_peer),
        program_symbols=(source_address,),
        buffer_abi=(source_abi,),
    )
    destination_fragment = _fragment(
        source,
        kind=FragmentKind.MOE_TRANSFER,
        claims=(recv.id, wait.id),
        stream=CoreFragmentStream(
            destination_core,
            (recv_record, wait_record),
            tuple(sorted((
                RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
                RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
                RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, destination_peer.id),
                RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
            ), key=lambda item: (item.record_index, _RUNTIME_FIELD_ORDER[item.field]))),
            (AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination_address.id, 0),),
        ),
        runtime_symbols=(fsm, token, destination_peer),
        program_symbols=(destination_address,),
        buffer_abi=(destination_abi,),
    )
    return source_fragment, destination_fragment


def lower_lite_moe_dp4_infer(
    intent: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
) -> tuple[CommandFragment, ...]:
    _indices(intent, source)
    fragments = [
        lower_lite_moe_dp4_state_load_unit(unit, intent, source)
        for unit in intent.state_loads
    ]
    fragments.extend(
        lower_lite_moe_dp4_compute_unit(unit, intent, source)
        for unit in intent.compute_units
    )
    for unit in intent.dte_units:
        fragments.extend(lower_lite_moe_dp4_dte_unit(unit, intent, source))
    result = tuple(sorted(fragments, key=lambda item: item.id))
    claims = tuple(action for fragment in result for action in fragment.claimed_action_ids)
    if (
        len(result) != 80
        or len(claims) != 92
        or set(claims) != {item.id for item in source.global_dag.actions}
    ):
        raise SchemaError("DP4 infer must cover 92 actions once in 80 leaves", path="fragments")
    return result


def validate_lite_moe_dp4_infer_fragment(
    fragment: CommandFragment,
    intent: LiteMoeDp4InferN6Intent,
    source: LiteMoeDp4ExecutionCase,
    path: str = "command_fragment",
) -> None:
    fragment.validate(path)
    candidates = tuple(
        item for item in lower_lite_moe_dp4_infer(intent, source)
        if item.claimed_action_ids == fragment.claimed_action_ids
    )
    if len(candidates) != 1 or fragment != candidates[0]:
        raise SchemaError("fragment is not exact DP4 infer lowering", path=path)


def build_lite_moe_dp4_tape_buffer_abis(
    source: LiteMoeDp4TrainForward,
) -> tuple[BufferABI, ...]:
    source.validate("source")
    graph = source.forward.n4.graph
    result = []
    for index, tape in enumerate(source.tape_buffers):
        die = next(item for item in graph.fabric.dies if item.id == tape.die_id)
        core = next(item for item in die.cores if item.id == tape.core_ref)
        profile = next(item for item in graph.fabric.sram_profiles if item.id == core.sram_profile_ref)
        region = next(item for item in profile.regions if item.name == "comm")
        semantic = {
            "schedule_id": source.id,
            "binding_id": tape.id,
            "value_id": tape.value_ref,
            "logical_core": LogicalCoreRef(tape.die_id, core.local_core_id),
            "tensor_slice": TensorSlice(tape.value_ref, (0, 0), (1, 32)),
            "region_ref": region.id,
            "region_offset_bytes": tape.address - region.base_bytes,
            "size_bytes": tape.size_bytes,
            "alignment_bytes": tape.alignment_bytes,
            "banks": tuple(sorted({
                ((tape.address + offset) // profile.bank_interleave_bytes)
                % profile.bank_count
                for offset in range(0, tape.size_bytes, profile.bank_interleave_bytes)
            })),
            "storage_id": f"moe.dp4.tape.storage.{tape.id}",
            "alias_of": None,
            "lifetime_start": tape.ordinal,
            "lifetime_end_exclusive": tape.ordinal + 1,
            "dtype": DType.FP16,
            "layout": "s3_lite_moe_dp4_tape",
            "ownership": BufferOwnership.OWNED,
        }
        abi = BufferABI(
            stable_artifact_id("buffer_abi", semantic, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION),
            **semantic,
        )
        abi.validate(f"tape_buffer_abis[{index}]")
        result.append(abi)
    return tuple(result)


def lower_lite_moe_dp4_tape_copy(
    unit: LiteMoeDp4TapeCopy,
    source: LiteMoeDp4TrainForward,
    infer_intent: LiteMoeDp4InferN6Intent,
    tape_buffer_abis: tuple[BufferABI, ...],
) -> CommandFragment:
    if type(unit) is not LiteMoeDp4TapeCopy or unit not in source.tape_copies:
        raise SchemaError("tape copy is not owned by training-forward", path="unit")
    unit.validate("unit")
    source.validate("source")
    infer_intent.validate("infer_intent")
    source_abi = next(
        (item for item in infer_intent.buffer_abis if item.binding_id == unit.source_buffer_ref),
        None,
    )
    destination = next(
        (item for item in tape_buffer_abis if item.binding_id == unit.destination_buffer_ref),
        None,
    )
    if (
        source_abi is None or destination is None
        or source_abi.logical_core != destination.logical_core
        or source_abi.size_bytes != unit.bytes
        or destination.size_bytes != unit.bytes
        or source_abi.dtype is not DType.FP16
        or destination.dtype is not DType.FP16
        or source_abi.logical_core.die_id != unit.die_id
    ):
        raise SchemaError("tape copy BufferABI closure mismatch", path="unit")
    graph = source.forward.n4.graph
    die = next(item for item in graph.fabric.dies if item.id == unit.die_id)
    core = next(item for item in die.cores if item.id == unit.core_ref)
    region = _region_symbol(destination.region_ref)
    label = _label_symbol(destination, core.runtime_core_id)
    src = _absolute_symbol(source_abi)
    dst = _absolute_symbol(destination)
    token = _runtime_symbol(RuntimeSymbolKind.DTE_TOKEN, unit.id, ("dp4_tape_copy", unit.id))
    alloc = RelocatableRecord(unit.id, RecordOpcode.SRAM_ALLOC_AT, (
        RecordOperand.address("region_name", SemanticOperandId.REGION_NAME, region.id),
        RecordOperand.address("label_symbol", SemanticOperandId.LABEL_SYMBOL, label.id),
        RecordOperand.literal("region_offset_bytes", destination.region_offset_bytes),
        RecordOperand.literal("size_bytes", destination.size_bytes),
        RecordOperand.literal("alignment_bytes", destination.alignment_bytes),
        RecordOperand.literal("lifetime", 0),
        RecordOperand.literal("spillable", False),
    ))
    issue = RelocatableRecord(unit.id, RecordOpcode.DTE_ISSUE, (
        RecordOperand.literal("direction", 0),
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
        RecordOperand.literal("payload_bits", unit.bytes * 8),
        RecordOperand.literal("size_bytes", unit.bytes),
        RecordOperand.literal("hbm_address", 0),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, src.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, dst.id),
    ))
    wait = RelocatableRecord(
        unit.id,
        RecordOpcode.DTE_WAIT,
        (RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),),
    )
    fragment = CommandFragment.create(
        producer_pass="lite_moe_dp4_train_forward_lowering",
        source_global_dag_id=source.id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(unit.id,),
        core_streams=(CoreFragmentStream(
            destination.logical_core,
            (alloc, issue, wait),
            (
                RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
                RuntimeRelocation(2, RuntimeOperandField.DTE_TOKEN, token.id),
            ),
            tuple(sorted((
                AddressRelocation(0, SemanticOperandId.REGION_NAME, ProgramSymbolKind.SRAM_REGION, region.id, 0),
                AddressRelocation(0, SemanticOperandId.LABEL_SYMBOL, ProgramSymbolKind.SRAM_LABEL, label.id, 0),
                AddressRelocation(1, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, src.id, 0),
                AddressRelocation(1, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, dst.id, 0),
            ), key=lambda item: (item.record_index, int(item.operand_id)))),
        ),),
        runtime_symbols=(token,),
        program_symbols=_unique_by_id((region, label, src, dst)),
        buffer_abi=tuple(sorted((source_abi, destination), key=lambda item: item.id)),
        state_abi=(),
    )
    fragment.validate("tape_fragment")
    return fragment


def lower_lite_moe_dp4_train_forward_tapes(
    source: LiteMoeDp4TrainForward,
    infer_intent: LiteMoeDp4InferN6Intent,
) -> tuple[tuple[BufferABI, ...], tuple[CommandFragment, ...]]:
    tape_abis = build_lite_moe_dp4_tape_buffer_abis(source)
    fragments = tuple(sorted((
        lower_lite_moe_dp4_tape_copy(unit, source, infer_intent, tape_abis)
        for unit in source.tape_copies
    ), key=lambda item: item.id))
    if len(fragments) != 8 or {item.claimed_action_ids[0] for item in fragments} != {
        item.id for item in source.tape_copies
    }:
        raise SchemaError("tape lowering must cover eight copies once", path="tape_fragments")
    return tape_abis, fragments


def rebase_lite_moe_dp4_infer_fragments(
    fragments: tuple[CommandFragment, ...],
    source_global_dag_id: str,
) -> tuple[CommandFragment, ...]:
    """Change only the top DAG provenance so infer leaves share one TF artifact."""

    result = tuple(sorted((
        CommandFragment.create(
            producer_pass=fragment.producer_pass,
            source_global_dag_id=source_global_dag_id,
            kind=fragment.kind,
            claimed_action_ids=fragment.claimed_action_ids,
            core_streams=fragment.core_streams,
            runtime_symbols=fragment.runtime_symbols,
            program_symbols=fragment.program_symbols,
            buffer_abi=fragment.buffer_abi,
            state_abi=fragment.state_abi,
        )
        for fragment in fragments
    ), key=lambda item: item.id))
    if len(result) != len(fragments) or any(
        (
            rebased.producer_pass,
            rebased.kind,
            rebased.claimed_action_ids,
            rebased.core_streams,
            rebased.runtime_symbols,
            rebased.program_symbols,
            rebased.buffer_abi,
            rebased.state_abi,
        ) != (
            original.producer_pass,
            original.kind,
            original.claimed_action_ids,
            original.core_streams,
            original.runtime_symbols,
            original.program_symbols,
            original.buffer_abi,
            original.state_abi,
        )
        for rebased in result
        for original in fragments
        if original.claimed_action_ids == rebased.claimed_action_ids
    ):
        raise SchemaError("infer fragment rebase changed leaf semantics", path="fragments")
    return result


__all__ = [
    "lower_lite_moe_dp4_compute_unit",
    "lower_lite_moe_dp4_dte_unit",
    "lower_lite_moe_dp4_infer",
    "lower_lite_moe_dp4_state_load_unit",
    "validate_lite_moe_dp4_infer_fragment",
    "build_lite_moe_dp4_tape_buffer_abis",
    "lower_lite_moe_dp4_tape_copy",
    "lower_lite_moe_dp4_train_forward_tapes",
    "rebase_lite_moe_dp4_infer_fragments",
]
