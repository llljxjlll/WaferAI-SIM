from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    OperandKind,
    ProgramSymbol,
    ProgramSymbolDefinition,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RegionManifest,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    RuntimeSymbol,
    RuntimeSymbolKind,
    SemanticOperandId,
    PlanBarrierEventPhase,
    canonical_plan_barrier_core_symbol,
    canonical_plan_barrier_event_symbol,
)
from llm.frontend.wafer_frontend.schema.action import BarrierScope, ComputeOperand
from llm.frontend.wafer_frontend.schema.global_action import (
    ActionBufferUse,
    GlobalAction,
    GlobalActionDAG,
    LogicalCoreRef,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import (
    GemmPartition,
    GemmWorkload,
    OpKind,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferOwnership,
    BufferUseRole,
    IR2ProjectionResult,
    IntraDieScheduleSet,
    SemanticTaskKind,
    TensorSlice,
    dense_row_major_view_byte_addend,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json, loads_dataclass

from test_global_action_schema import (
    _create_global,
    _two_by_two_case,
    _two_die_case,
    _with_recv_wait,
)
from test_ir2_schema import valid_dag, valid_reduce_case, valid_schedule
from test_naive_intra_die import _complete_projection
from _fixtures import valid_ir1
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy


def _fixture_buffer_abis(actions: tuple[GlobalAction, ...]) -> tuple[BufferABI, ...]:
    grouped = {}
    for action in actions:
        for use in action.buffer_uses:
            grouped.setdefault(
                (action.source.schedule_id, use.binding_id), []
            ).append((action, use))
    result = []
    for (schedule_id, binding_id), entries in grouped.items():
        first_action, first_use = entries[0]
        offsets = tuple(
            min(use.tensor_slice.offset[axis] for _action, use in entries)
            for axis in range(len(first_use.tensor_slice.shape))
        )
        ends = tuple(
            max(
                use.tensor_slice.offset[axis] + use.tensor_slice.shape[axis]
                for _action, use in entries
            )
            for axis in range(len(first_use.tensor_slice.shape))
        )
        root = TensorSlice(
            first_use.tensor_slice.value_id,
            offsets,
            tuple(end - offset for offset, end in zip(offsets, ends)),
        )
        dtype = first_action.dtype or DType.FP16
        element_bytes = 2 if dtype is DType.FP16 else 4
        elements = 1
        for extent in root.shape:
            elements *= extent
        assert first_action.logical_core is not None
        result.append(
            BufferABI(
                f"fixture_abi_{binding_id}",
                schedule_id,
                binding_id,
                root.value_id,
                first_action.logical_core,
                root,
                "fixture_sram",
                0,
                elements * element_bytes,
                element_bytes,
                (),
                f"fixture_storage_{binding_id}",
                None,
                min(action.core_order_index for action, _use in entries),
                max(action.core_order_index for action, _use in entries) + 1,
                dtype,
                "row_major",
                (
                    BufferOwnership.OWNED
                    if any(use.access is BufferAccess.WRITE for _action, use in entries)
                    else BufferOwnership.BORROWED
                ),
            )
        )
    return tuple(sorted(result, key=lambda abi: abi.id))


def _fixture_relocation_addend(
    action: GlobalAction,
    relocation: AddressRelocation,
    abis: tuple[BufferABI, ...],
) -> int:
    role = (
        BufferUseRole.SEND_SOURCE
        if relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS
        and action.task_kind is SemanticTaskKind.SEND
        else BufferUseRole.RECV_DESTINATION
    )
    use = next(use for use in action.buffer_uses if use.role is role)
    abi = next(abi for abi in abis if abi.binding_id == use.binding_id)
    delta = dense_row_major_view_byte_addend(
        abi.tensor_slice, use.tensor_slice, abi.dtype
    )
    return abi.region_offset_bytes + delta


def _send_record(action_id: str, stem: str):
    runtime = (
        RuntimeSymbol(f"{stem}_fsm", RuntimeSymbolKind.DTE_FSM, action_id),
        RuntimeSymbol(f"{stem}_peer", RuntimeSymbolKind.RUNTIME_CORE, action_id),
    )
    program = ProgramSymbol(f"{stem}_source", ProgramSymbolKind.SRAM_REGION, action_id)
    record = RelocatableRecord(action_id, RecordOpcode.DTE_SEND, (
        RecordOperand.literal("mode", 0), RecordOperand.literal("source_space", 0),
        RecordOperand.literal("completion", 1), RecordOperand.literal("datatype", 0),
        RecordOperand.literal("reduce_op", 0),
        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, f"{stem}_fsm"),
        RecordOperand.literal("token", 0),
        RecordOperand.literal("length_bytes", 32),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, program.id),
        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, f"{stem}_peer"),
        RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
        RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
        RecordOperand.literal("epoch", 0),
    ))
    runtime_relocs = (
        RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, f"{stem}_fsm"),
        RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, f"{stem}_peer"),
    )
    address_relocs = (AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.SRAM_REGION, program.id, 0),)
    return record, runtime, (program,), runtime_relocs, address_relocs


def _recv_record(action_id: str, stem: str):
    runtime = (
        RuntimeSymbol(f"{stem}_fsm", RuntimeSymbolKind.DTE_FSM, action_id),
        RuntimeSymbol(f"{stem}_peer", RuntimeSymbolKind.RUNTIME_CORE, action_id),
    )
    program = ProgramSymbol(f"{stem}_destination", ProgramSymbolKind.SRAM_REGION, action_id)
    record = RelocatableRecord(action_id, RecordOpcode.DTE_RECV, (
        RecordOperand.literal("mode", 0), RecordOperand.literal("completion", 1),
        RecordOperand.literal("datatype", 0), RecordOperand.literal("reduce_op", 0),
        RecordOperand.runtime("fsm_id", RuntimeOperandField.DTE_FSM, f"{stem}_fsm"),
        RecordOperand.literal("token", 0),
        RecordOperand.literal("length_bytes", 32),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, program.id),
        RecordOperand.runtime("peer_core", RuntimeOperandField.PEER_CORE, f"{stem}_peer"),
        RecordOperand.literal("expected_sources", 0), RecordOperand.literal("tree_id", 0),
        RecordOperand.literal("group_id", 0), RecordOperand.literal("collective_id", 0),
        RecordOperand.literal("epoch", 0),
    ))
    runtime_relocs = (
        RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, f"{stem}_fsm"),
        RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, f"{stem}_peer"),
    )
    address_relocs = (AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.SRAM_REGION, program.id, 0),)
    return record, runtime, (program,), runtime_relocs, address_relocs


def _async_recv_record(action: GlobalAction, token: RuntimeSymbol):
    record, runtime, program, runtime_relocs, address_relocs = _recv_record(
        action.id, action.id
    )
    operands = list(record.operands)
    operands[1] = RecordOperand.literal("completion", 0)
    operands[5] = RecordOperand.runtime(
        "token", RuntimeOperandField.DTE_TOKEN, token.id
    )
    return (
        replace(record, operands=tuple(operands)),
        runtime + (token,),
        program,
        runtime_relocs
        + (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),),
        address_relocs,
    )


def _shared_wait_record(action: GlobalAction, token: RuntimeSymbol):
    record = RelocatableRecord(
        action.id,
        RecordOpcode.DTE_WAIT,
        (
            RecordOperand.runtime(
                "token", RuntimeOperandField.DTE_TOKEN, token.id
            ),
        ),
    )
    return (
        record,
        (),
        (),
        (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),),
        (),
    )


def _wait_record(action_id: str, stem: str):
    symbol = RuntimeSymbol(f"{stem}_token", RuntimeSymbolKind.DTE_TOKEN, action_id)
    record = RelocatableRecord(action_id, RecordOpcode.DTE_WAIT, (
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, symbol.id),
    ))
    return record, (symbol,), (), (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, symbol.id),), ()


def _compute_record(action_id: str, opcode: RecordOpcode, parameters: tuple[int, ...]):
    addressed_data = opcode in (RecordOpcode.MATMUL, RecordOpcode.RESIDUAL)
    symbols = tuple(sorted((
        ProgramSymbol(f"{action_id}_input", ProgramSymbolKind.ABSOLUTE_ADDRESS, action_id),
        *(
            (ProgramSymbol(
                f"{action_id}_data",
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                action_id,
            ),)
            if addressed_data
            else ()
        ),
        ProgramSymbol(f"{action_id}_output", ProgramSymbolKind.ABSOLUTE_ADDRESS, action_id),
    ), key=lambda item: item.id))
    by_suffix = {symbol.id.rsplit("_", 1)[-1]: symbol for symbol in symbols}
    record = RelocatableRecord(action_id, opcode, (
        RecordOperand.literal("datatype", 1),
        RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS, by_suffix["input"].id),
        (
            RecordOperand.address(
                "data_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                by_suffix["data"].id,
            )
            if addressed_data
            else RecordOperand.literal("data_address", 0)
        ),
        RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, by_suffix["output"].id),
        RecordOperand.literal("parameters", parameters),
    ))
    relocations = (
        AddressRelocation(0, SemanticOperandId.COMPUTE_INPUT_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, by_suffix["input"].id, 0),
        *(
            (AddressRelocation(
                0,
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                by_suffix["data"].id,
                0,
            ),)
            if addressed_data
            else ()
        ),
        AddressRelocation(0, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, by_suffix["output"].id, 0),
    )
    return record, symbols, relocations


_BIND_INPUT_COUNT_BY_COMPUTE_OPCODE = {
    RecordOpcode.MATMUL: 1,
    RecordOpcode.ATTENTION: 1,
    RecordOpcode.SWIGLU: 1,
    RecordOpcode.RESIDUAL: 2,
    RecordOpcode.RMSNORM: 1,
}


def _sram_bind_record(action_id: str, input_count: int):
    input_ids = tuple(
        SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0) + index)
        for index in range(input_count)
    )
    input_symbols = tuple(
        ProgramSymbol(
            f"{action_id}_bind_input_{index}",
            ProgramSymbolKind.SRAM_LABEL,
            action_id,
        )
        for index in range(input_count)
    )
    output_symbol = ProgramSymbol(
        f"{action_id}_bind_output",
        ProgramSymbolKind.SRAM_LABEL,
        action_id,
    )
    operands = (
        RecordOperand.literal("input_count", input_count),
        *(
            RecordOperand.address(
                f"input_label_{index}", input_ids[index], input_symbols[index].id
            )
            if index < input_count
            else RecordOperand.literal(f"input_label_{index}", 0)
            for index in range(16)
        ),
        RecordOperand.address(
            "output_label",
            SemanticOperandId.SRAM_BIND_OUTPUT,
            output_symbol.id,
        ),
    )
    relocations = tuple(
        AddressRelocation(
            0, operand_id, ProgramSymbolKind.SRAM_LABEL, symbol.id, 0
        )
        for operand_id, symbol in zip(input_ids, input_symbols)
    ) + (
        AddressRelocation(
            0,
            SemanticOperandId.SRAM_BIND_OUTPUT,
            ProgramSymbolKind.SRAM_LABEL,
            output_symbol.id,
            0,
        ),
    )
    return (
        RelocatableRecord(action_id, RecordOpcode.SRAM_BIND, operands),
        tuple(sorted((*input_symbols, output_symbol), key=lambda item: item.id)),
        relocations,
    )


def _compute_fragment(dag: GlobalActionDAG, action: GlobalAction, opcode: RecordOpcode, parameters: tuple[int, ...]) -> CommandFragment:
    bind, label_symbols, label_relocations = _sram_bind_record(
        action.id, _BIND_INPUT_COUNT_BY_COMPUTE_OPCODE[opcode]
    )
    record, symbols, relocations = _compute_record(action.id, opcode, parameters)
    return CommandFragment.create(
        producer_pass="compute_fixture", source_global_dag_id=dag.id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            action.logical_core,
            (bind, record),
            (),
            label_relocations
            + tuple(replace(item, record_index=1) for item in relocations),
        ),),
        runtime_symbols=(),
        program_symbols=tuple(sorted(label_symbols + symbols, key=lambda item: item.id)),
        buffer_abi=_fixture_buffer_abis((action,)),
    )


def _reduce_fragment_case() -> tuple[GlobalActionDAG, GlobalAction, CommandFragment]:
    ir1, source_dag, schedule = valid_reduce_case()
    projection = IR2ProjectionResult.create(
        producer_pass="reduce_fragment_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=("fp_reduce",),
        standalone_collective_plan_ids=(),
        dags=(source_dag,),
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="reduce_fragment_schedule_fixture",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=(schedule,),
    )
    dag = _create_global(ir1, projection, schedule_set)
    action = dag.actions[0]
    source = ProgramSymbol(
        "reduce_fragment_source", ProgramSymbolKind.ABSOLUTE_ADDRESS, action.id
    )
    destination = ProgramSymbol(
        "reduce_fragment_destination",
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        action.id,
    )
    record = RelocatableRecord(
        action.id,
        RecordOpcode.LOCAL_REDUCE,
        (
            RecordOperand.literal("input_dtype", 0),
            RecordOperand.literal("accumulator_dtype", 1),
            RecordOperand.literal("output_dtype", 0),
            RecordOperand.literal("reduce_op", 1),
            RecordOperand.literal("rounding", 0),
            RecordOperand.literal("order", 0),
            RecordOperand.literal("input_count", 2),
            RecordOperand.literal("element_count", 16),
            RecordOperand.literal("input_stride_bytes", 32),
            RecordOperand.address(
                "source_address", SemanticOperandId.SOURCE_ADDRESS, source.id
            ),
            RecordOperand.address(
                "destination_address",
                SemanticOperandId.DESTINATION_ADDRESS,
                destination.id,
            ),
        ),
    )
    stream = CoreFragmentStream(
        action.logical_core,
        (record,),
        (),
        (
            AddressRelocation(
                0,
                SemanticOperandId.SOURCE_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                source.id,
                0,
            ),
            AddressRelocation(
                0,
                SemanticOperandId.DESTINATION_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                destination.id,
                0,
            ),
        ),
    )
    fragment = CommandFragment.create(
        producer_pass="reduce_fragment_fixture",
        source_global_dag_id=dag.id,
        kind=FragmentKind.ISA_REGION,
        claimed_action_ids=(action.id,),
        core_streams=(stream,),
        runtime_symbols=(),
        program_symbols=tuple(
            sorted((source, destination), key=lambda item: item.id)
        ),
        buffer_abi=_fixture_buffer_abis((action,)),
    )
    fragment.validate_against(dag)
    return dag, action, fragment


def _single_action_dag(template: GlobalActionDAG, action: GlobalAction) -> GlobalActionDAG:
    return GlobalActionDAG.create(
        producer_pass=template.producer_pass,
        source_ir1_id=template.source_ir1_id,
        source_state_manifest_id=template.source_state_manifest_id,
        source_projection_id=template.source_projection_id,
        source_schedule_set_id=template.source_schedule_set_id,
        scheduled_dags=template.scheduled_dags,
        actions=(action,),
    )


def _matmul_fragment_case():
    ir1 = valid_ir1()
    source_dag = valid_dag()
    schedule = valid_schedule(source_dag)
    projection = IR2ProjectionResult.create(
        producer_pass="bind_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=("fp_0",),
        standalone_collective_plan_ids=(),
        dags=(source_dag,),
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="bind_schedule_fixture",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=(schedule,),
    )
    template = _create_global(ir1, projection, schedule_set)
    base = next(
        action
        for action in template.actions
        if action.task_kind is SemanticTaskKind.COMP
    )
    input_shape = base.buffer_uses[0].tensor_slice.shape
    output_shape = base.buffer_uses[-1].tensor_slice.shape
    gemm = GemmWorkload(
        (input_shape[0], output_shape[-1], input_shape[-1]),
        (input_shape[0], output_shape[-1], input_shape[-1]),
        GemmPartition.REPLICATED,
        DType.FP16,
    )
    compute = replace(
        base.compute,
        op_kind=OpKind.GEMM,
        workload=gemm,
        impl_ref="matmul_forward",
        inputs=(
            ComputeOperand(base.compute.inputs[0].value_id, "lhs"),
            ComputeOperand("fixture_weight", "rhs"),
        ),
        outputs=(
            ComputeOperand(base.compute.outputs[0].value_id, "output"),
        ),
    )
    action = replace(
        base,
        op_kind=OpKind.GEMM,
        read_values=base.read_values + ("fixture_weight",),
        compute=compute,
        buffer_uses=(
            base.buffer_uses[0],
            ActionBufferUse(
                "fixture_weight_binding",
                BufferAccess.READ,
                BufferUseRole.COMP_INPUT,
                1,
                None,
                replace(
                    base.buffer_uses[0].tensor_slice,
                    value_id="fixture_weight",
                ),
            ),
            base.buffer_uses[1],
        ),
    )
    dag = _single_action_dag(template, action)
    rank_m, rank_n, rank_k = compute.workload.rank_shape
    fragment = _compute_fragment(
        dag,
        action,
        RecordOpcode.MATMUL,
        (1, rank_m, rank_k, rank_n),
    )
    fragment.validate_against(dag)
    return dag, action, fragment


def _local_copy_fragment(dag: GlobalActionDAG, action: GlobalAction) -> CommandFragment:
    token = RuntimeSymbol(f"{action.id}_copy_token", RuntimeSymbolKind.DTE_TOKEN, action.id)
    source = ProgramSymbol(f"{action.id}_copy_source", ProgramSymbolKind.SRAM_REGION, action.id)
    destination = ProgramSymbol(f"{action.id}_copy_destination", ProgramSymbolKind.SRAM_REGION, action.id)
    issue = RelocatableRecord(action.id, RecordOpcode.DTE_ISSUE, (
        RecordOperand.literal("direction", 0),
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
        RecordOperand.literal("payload_bits", action.bytes * 8), RecordOperand.literal("size_bytes", action.bytes),
        RecordOperand.literal("hbm_address", 0),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination.id),
    ))
    wait = RelocatableRecord(action.id, RecordOpcode.DTE_WAIT, (
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
    ))
    stream = CoreFragmentStream(
        action.logical_core, (issue, wait),
        (
            RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
            RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id),
        ),
        (
            AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.SRAM_REGION, source.id, 0),
            AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.SRAM_REGION, destination.id, 0),
        ),
    )
    return CommandFragment.create(
        producer_pass="copy_fixture", source_global_dag_id=dag.id,
        kind=FragmentKind.ISA_REGION, claimed_action_ids=(action.id,),
        core_streams=(stream,), runtime_symbols=(token,),
        program_symbols=tuple(sorted((source, destination), key=lambda item: item.id)),
        buffer_abi=_fixture_buffer_abis((action,)),
    )


def _record_for(
    action: GlobalAction,
    async_tokens: dict[str, RuntimeSymbol],
    recv_by_wait: dict[str, str],
):
    stem = action.id
    if action.task_kind is SemanticTaskKind.SEND:
        return _send_record(action.id, stem)
    if action.task_kind is SemanticTaskKind.RECV:
        if action.id in async_tokens:
            return _async_recv_record(action, async_tokens[action.id])
        return _recv_record(action.id, stem)
    if action.task_kind is SemanticTaskKind.WAIT:
        recv_id = recv_by_wait[action.id]
        return _shared_wait_record(action, async_tokens[recv_id])
    raise AssertionError(action.task_kind)


def _fragment(
    dag: GlobalActionDAG,
    actions: tuple[GlobalAction, ...] | None = None,
    *,
    action_order: tuple[str, ...] | None = None,
) -> CommandFragment:
    selected = actions or tuple(action for action in dag.actions if action.task_kind is not SemanticTaskKind.TRANSIT)
    all_actions = {action.id: action for action in dag.actions}
    recv_by_wait = {
        wait.id: dependency
        for wait in dag.actions
        if wait.task_kind is SemanticTaskKind.WAIT
        for dependency in wait.deps
        if dependency in all_actions
        and all_actions[dependency].task_kind is SemanticTaskKind.RECV
        and all_actions[dependency].sync is not None
        and wait.sync is not None
        and all_actions[dependency].sync.completion_event == wait.sync.wait_event
    }
    async_tokens = {
        recv_id: RuntimeSymbol(
            f"{recv_id}_async_token",
            RuntimeSymbolKind.DTE_TOKEN,
            all_actions[recv_id].runtime_binding.token_symbol,
        )
        for recv_id in recv_by_wait.values()
    }
    action_index = {action.id: action for action in selected}
    if action_order is None:
        ordered = sorted(selected, key=lambda action: (action.logical_core.die_id, action.logical_core.local_core_id, action.core_order_index))
    else:
        ordered = [action_index[action_id] for action_id in action_order]
    grouped: dict[LogicalCoreRef, list[GlobalAction]] = {}
    for action in ordered:
        grouped.setdefault(action.logical_core, []).append(action)
    fixture_abis = _fixture_buffer_abis(tuple(selected))
    streams = []
    runtime_symbols: list[RuntimeSymbol] = []
    program_symbols: list[ProgramSymbol] = []
    for core in sorted(grouped, key=lambda item: (item.die_id, item.local_core_id)):
        records = []
        runtime_relocs = []
        address_relocs = []
        for action in grouped[core]:
            record, runtime, program, record_runtime, record_address = _record_for(
                action, async_tokens, recv_by_wait
            )
            record_index = len(records)
            records.append(record)
            runtime_symbols.extend(runtime)
            program_symbols.extend(program)
            runtime_relocs.extend(replace(item, record_index=record_index) for item in record_runtime)
            address_relocs.extend(
                replace(
                    item,
                    record_index=record_index,
                    addend=_fixture_relocation_addend(
                        action, item, fixture_abis
                    ),
                )
                for item in record_address
            )
        runtime_relocs.sort(key=lambda item: (item.record_index, list(RuntimeOperandField).index(item.field)))
        address_relocs.sort(key=lambda item: (item.record_index, int(item.operand_id)))
        streams.append(CoreFragmentStream(core, tuple(records), tuple(runtime_relocs), tuple(address_relocs)))
    return CommandFragment.create(
        producer_pass="fragment_fixture",
        source_global_dag_id=dag.id,
        kind=FragmentKind.ISA_REGION,
        claimed_action_ids=tuple(sorted(action.id for action in selected)),
        core_streams=tuple(streams),
        runtime_symbols=tuple(
            sorted(
                {symbol.id: symbol for symbol in runtime_symbols}.values(),
                key=lambda item: item.id,
            )
        ),
        program_symbols=tuple(sorted(program_symbols, key=lambda item: item.id)),
        buffer_abi=fixture_abis,
    )


def _recreate(fragment: CommandFragment, **changes: object) -> CommandFragment:
    fields = fragment._semantic_key()
    fields.update(changes)
    return CommandFragment.create(producer_pass=fragment.producer_pass, **fields)


def _event_relocations(records: tuple[RelocatableRecord, ...]):
    relocations = tuple(
        RuntimeRelocation(record_index, operand.runtime_field, operand.symbol_ref)
        for record_index, record in enumerate(records)
        for operand in record.operands
        if operand.kind is OperandKind.RUNTIME_SYMBOL
    )
    return tuple(
        sorted(
            relocations,
            key=lambda item: (
                item.record_index,
                list(RuntimeOperandField).index(item.field),
            ),
        )
    )


def _plan_barrier_fragment(tp: int, *, barrier_only: bool = False):
    ir1, projection = _complete_projection(tp=tp, large_sram=True)
    schedule_set = NaiveIntraDiePolicy().schedule(projection, ir1)
    dag = _create_global(ir1, projection, schedule_set)
    first = next(
        action
        for action in dag.actions
        if action.task_kind is SemanticTaskKind.BARRIER
    )
    barrier = first.sync.barrier
    by_rank = {
        action.origin_ref.rank: action
        for action in dag.actions
        if action.task_kind is SemanticTaskKind.BARRIER
        and action.sync.barrier.id == barrier.id
    }
    participants = tuple(by_rank[rank] for rank in barrier.participant_ranks)
    if barrier_only:
        participants = tuple(replace(action, deps=()) for action in participants)
        dag = GlobalActionDAG.create(
            producer_pass="barrier_only_fixture",
            source_ir1_id=dag.source_ir1_id,
            source_state_manifest_id=dag.source_state_manifest_id,
            source_projection_id=dag.source_projection_id,
            source_schedule_set_id=dag.source_schedule_set_id,
            scheduled_dags=dag.scheduled_dags,
            actions=participants,
        )
    leader, *peers = participants
    records_by_action: dict[str, tuple[RelocatableRecord, ...]] = {}
    runtime_symbols = {
        symbol.id: symbol
        for symbol in (
            canonical_plan_barrier_core_symbol(dag.id, action)
            for action in participants
        )
    }

    def event_record(
        owner: GlobalAction,
        opcode: RecordOpcode,
        phase: PlanBarrierEventPhase,
        source: GlobalAction,
        destination: GlobalAction,
    ) -> RelocatableRecord:
        source_core = canonical_plan_barrier_core_symbol(dag.id, source)
        destination_core = canonical_plan_barrier_core_symbol(dag.id, destination)
        event = canonical_plan_barrier_event_symbol(
            dag.id, phase, source, destination
        )
        runtime_symbols[event.id] = event
        operands = (
            RecordOperand.runtime(
                "source_core", RuntimeOperandField.SOURCE_CORE, source_core.id
            ),
            RecordOperand.runtime(
                "destination_core",
                RuntimeOperandField.DESTINATION_CORE,
                destination_core.id,
            ),
            RecordOperand.runtime(
                "tag", RuntimeOperandField.EVENT_TAG, event.id
            ),
        )
        if opcode is RecordOpcode.EVENT_WAIT:
            operands = (*operands, RecordOperand.literal("count", 1))
        return RelocatableRecord(owner.id, opcode, operands)

    records_by_action[leader.id] = (
        *(
            event_record(
                leader,
                RecordOpcode.EVENT_WAIT,
                PlanBarrierEventPhase.ARRIVE,
                peer,
                leader,
            )
            for peer in peers
        ),
        *(
            event_record(
                leader,
                RecordOpcode.EVENT_SET,
                PlanBarrierEventPhase.RELEASE,
                leader,
                peer,
            )
            for peer in peers
        ),
    )
    for peer in peers:
        records_by_action[peer.id] = (
            event_record(
                peer,
                RecordOpcode.EVENT_SET,
                PlanBarrierEventPhase.ARRIVE,
                peer,
                leader,
            ),
            event_record(
                peer,
                RecordOpcode.EVENT_WAIT,
                PlanBarrierEventPhase.RELEASE,
                leader,
                peer,
            ),
        )
    streams = tuple(
        CoreFragmentStream(
            action.logical_core,
            records_by_action[action.id],
            _event_relocations(records_by_action[action.id]),
            (),
        )
        for action in sorted(
            participants,
            key=lambda item: (
                item.logical_core.die_id,
                item.logical_core.local_core_id,
            ),
        )
    )
    fragment = CommandFragment.create(
        producer_pass="plan_barrier_fixture",
        source_global_dag_id=dag.id,
        kind=FragmentKind.STANDALONE_COLLECTIVE,
        claimed_action_ids=tuple(sorted(action.id for action in participants)),
        core_streams=streams,
        runtime_symbols=tuple(sorted(runtime_symbols.values(), key=lambda item: item.id)),
        program_symbols=(),
        buffer_abi=(),
    )
    return dag, participants, fragment


def _two_die_fragment():
    ir1, projection, schedule_set = _two_die_case()
    dag = _create_global(ir1, projection, schedule_set)
    fragment = _fragment(dag)
    fragment.validate_against(dag)
    return ir1, projection, schedule_set, dag, fragment


class ArtifactManifestSchemaTest(unittest.TestCase):
    def test_sram_bind_pair_arity_slots_labels_and_matmul_parameters_are_exact(self) -> None:
        dag, action, fragment = _matmul_fragment_case()
        stream = fragment.core_streams[0]
        bind, compute = stream.records
        label_relocations = tuple(
            item
            for item in stream.address_relocations
            if item.record_index == 0
        )
        compute_relocations = tuple(
            item
            for item in stream.address_relocations
            if item.record_index == 1
        )
        self.assertEqual(int(RecordOpcode.SRAM_BIND), 0x84)
        self.assertEqual(bind.operands[0].literal_value, 1)
        self.assertEqual(bind.operands[2].literal_value, 0)

        compute_only_ids = {item.symbol_ref for item in compute_relocations}
        missing = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    records=(compute,),
                    address_relocations=tuple(
                        replace(item, record_index=0)
                        for item in compute_relocations
                    ),
                ),
            ),
            program_symbols=tuple(
                item
                for item in fragment.program_symbols
                if item.id in compute_only_ids
            ),
        )
        with self.assertRaisesRegex(SchemaError, "contiguous SRAM_BIND"):
            missing.validate_against(dag)

        doubled = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    records=(bind, bind, compute),
                    address_relocations=(
                        *label_relocations,
                        *(replace(item, record_index=1) for item in label_relocations),
                        *(replace(item, record_index=2) for item in compute_relocations),
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "contiguous SRAM_BIND"):
            doubled.validate_against(dag)

        reversed_pair = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    records=(compute, bind),
                    address_relocations=(
                        *(replace(item, record_index=0) for item in compute_relocations),
                        *(replace(item, record_index=1) for item in label_relocations),
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "contiguous SRAM_BIND"):
            reversed_pair.validate_against(dag)

        bind_operands = list(bind.operands)
        zero_count = list(bind.operands)
        zero_count[0] = RecordOperand.literal("input_count", 0)
        with self.assertRaisesRegex(SchemaError, r"\[1,16\]"):
            replace(bind, operands=tuple(zero_count)).validate("bind")

        bind_operands[0] = RecordOperand.literal("input_count", 2)
        second_label = ProgramSymbol(
            f"{action.id}_bind_input_1",
            ProgramSymbolKind.SRAM_LABEL,
            action.id,
        )
        bind_operands[2] = RecordOperand.address(
            "input_label_1",
            SemanticOperandId.SRAM_BIND_INPUT_1,
            second_label.id,
        )
        input_label_relocation = next(
            item
            for item in label_relocations
            if item.operand_id is SemanticOperandId.SRAM_BIND_INPUT_0
        )
        arity = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    records=(replace(bind, operands=tuple(bind_operands)), compute),
                    address_relocations=tuple(
                        sorted(
                            stream.address_relocations
                            + (
                                AddressRelocation(
                                    0,
                                    SemanticOperandId.SRAM_BIND_INPUT_1,
                                    ProgramSymbolKind.SRAM_LABEL,
                                    second_label.id,
                                    0,
                                ),
                            ),
                            key=lambda item: (
                                item.record_index, int(item.operand_id)
                            ),
                        )
                    ),
                ),
            ),
            program_symbols=tuple(
                sorted(
                    fragment.program_symbols + (second_label,),
                    key=lambda item: item.id,
                )
            ),
        )
        with self.assertRaisesRegex(SchemaError, "input_count must exactly"):
            arity.validate_against(dag)

        inactive = list(bind.operands)
        inactive[2] = RecordOperand.literal("input_label_1", 1)
        with self.assertRaisesRegex(SchemaError, "inactive SRAM_BIND"):
            replace(bind, operands=tuple(inactive)).validate("bind")

        missing_label_relocation = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    address_relocations=tuple(
                        item
                        for item in stream.address_relocations
                        if item != input_label_relocation
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly one relocation"):
            missing_label_relocation.validate()

        bad_symbol = next(
            item
            for item in fragment.program_symbols
            if item.id == input_label_relocation.symbol_ref
        )
        wrong_kind = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    address_relocations=tuple(
                        replace(
                            item, symbol_kind=ProgramSymbolKind.ABSOLUTE_ADDRESS
                        )
                        if item == input_label_relocation
                        else item
                        for item in stream.address_relocations
                    ),
                ),
            ),
            program_symbols=tuple(
                replace(item, kind=ProgramSymbolKind.ABSOLUTE_ADDRESS)
                if item == bad_symbol
                else item
                for item in fragment.program_symbols
            ),
        )
        with self.assertRaisesRegex(SchemaError, "illegal for opcode"):
            wrong_kind.validate()

        parameters = list(compute.operands)
        rank_m, rank_n, rank_k = action.compute.workload.rank_shape
        parameters[-1] = RecordOperand.literal(
            "parameters", (rank_m, 1, rank_k, rank_n)
        )
        wrong_parameters = _recreate(
            fragment,
            core_streams=(
                replace(
                    stream,
                    records=(bind, replace(compute, operands=tuple(parameters))),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "rank-local"):
            wrong_parameters.validate_against(dag)

        label_definition = ProgramSymbolDefinition(
            bad_symbol,
            "fixture_label",
            0,
            0,
            (action.logical_core,),
        )
        label_definition.validate("label_definition")
        with self.assertRaisesRegex(SchemaError, "no physical address"):
            replace(label_definition, value=1).validate("label_definition")

    def test_local_reduce_backend_literals_and_payload_are_exact(self) -> None:
        dag, action, fragment = _reduce_fragment_case()
        stream = fragment.core_streams[0]
        record = stream.records[0]

        def with_literal(name: str, value: int) -> CommandFragment:
            changed = replace(
                record,
                operands=tuple(
                    replace(operand, literal_value=value)
                    if operand.name == name
                    else operand
                    for operand in record.operands
                ),
            )
            return _recreate(
                fragment,
                core_streams=(replace(stream, records=(changed,)),),
            )

        for name, value in (
            ("input_dtype", 1),
            ("accumulator_dtype", 0),
            ("output_dtype", 1),
            ("reduce_op", 0),
            ("rounding", 1),
            ("order", 1),
        ):
            with self.subTest(field=name):
                with self.assertRaisesRegex(SchemaError, "fixed FP16/FP32/FP16"):
                    with_literal(name, value).validate_against(dag)

        for input_count in (0, 1 << 16):
            with self.subTest(input_count=input_count):
                with self.assertRaisesRegex(SchemaError, "non-zero u16"):
                    with_literal("input_count", input_count).validate_against(dag)

        with self.assertRaisesRegex(SchemaError, "exact non-zero element_count"):
            with_literal("element_count", 15).validate_against(dag)
        with self.assertRaisesRegex(SchemaError, "exactly equal action bytes"):
            with_literal("input_stride_bytes", 16).validate_against(dag)

        odd_action = replace(action, bytes=31)
        odd_dag = _single_action_dag(dag, odd_action)
        odd_fragment = _recreate(fragment, source_global_dag_id=odd_dag.id)
        with self.assertRaisesRegex(SchemaError, "positive even FP16 bytes"):
            odd_fragment.validate_against(odd_dag)

    def test_dense_compute_impl_opcode_and_parameter_count_are_exact(self) -> None:
        from llm.frontend.wafer_frontend.schema.artifact_manifest import (
            _compute_record_abi,
        )
        from test_stage2_dense_forward_graph import _schedule_tiny

        _p, _r, _c, _s, bundle = _schedule_tiny(1)
        context = bundle.entries[0].lowering_context()
        actions = {
            action.compute.impl_ref: action.compute
            for action in context.global_dag.actions
            if action.compute is not None
        }
        expected = {
            "embedding_lookup": RecordOpcode.EMBEDDING_LOOKUP,
            "rms_norm": RecordOpcode.RMSNORM,
            "matmul_forward": RecordOpcode.MATMUL,
            "rope_qk_exact": RecordOpcode.ROPE_QK_EXACT,
            "attention_forward": RecordOpcode.ATTENTION_EXACT,
            "residual": RecordOpcode.RESIDUAL,
            "swiglu": RecordOpcode.SWIGLU,
        }
        self.assertEqual(set(actions), set(expected))
        for impl_ref, opcode in expected.items():
            with self.subTest(impl_ref=impl_ref):
                abi = _compute_record_abi(actions[impl_ref], path="compute")
                self.assertEqual(abi.opcode, opcode)
                if opcode in (
                    RecordOpcode.ROPE_QK_EXACT,
                    RecordOpcode.ATTENTION_EXACT,
                    RecordOpcode.EMBEDDING_LOOKUP,
                ):
                    self.assertEqual(abi.parameters, ())
        self.assertEqual(
            _compute_record_abi(actions["rms_norm"], path="compute").data_input_index,
            1,
        )
        return
    def test_local_copy_requires_issue_wait_shared_token_and_full_relocations(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        template = _create_global(ir1, projection, schedule_set)
        send = next(action for action in template.actions if action.task_kind is SemanticTaskKind.SEND)
        source_use = next(
            use
            for use in send.buffer_uses
            if use.role is BufferUseRole.SEND_SOURCE
        )
        destination_slice = replace(
            source_use.tensor_slice, value_id="copy_output"
        )
        copy = replace(
            send,
            task_kind=SemanticTaskKind.LOCAL_COPY,
            flow_id=None,
            source_rank=None,
            destination_rank=None,
            read_values=(source_use.tensor_slice.value_id,),
            write_values=("copy_output",),
            flow=None,
            flow_route=None,
            runtime_binding=None,
            buffer_uses=(
                replace(
                    source_use,
                    role=BufferUseRole.LOCAL_COPY_SOURCE,
                ),
                ActionBufferUse(
                    "copy_destination_binding",
                    BufferAccess.WRITE,
                    BufferUseRole.LOCAL_COPY_DESTINATION,
                    0,
                    None,
                    destination_slice,
                ),
            ),
        )
        dag = _single_action_dag(template, copy)
        fragment = _local_copy_fragment(dag, copy)
        fragment.validate_against(dag)
        stream = fragment.core_streams[0]

        with self.assertRaisesRegex(SchemaError, "exactly one relocation"):
            _recreate(fragment, core_streams=(replace(stream, address_relocations=stream.address_relocations[:-1]),)).validate()

        issue_only = replace(
            stream,
            records=(stream.records[0],),
            runtime_relocations=(stream.runtime_relocations[0],),
        )
        with self.assertRaisesRegex(SchemaError, "DTE_ISSUE then DTE_WAIT"):
            _recreate(fragment, core_streams=(issue_only,)).validate_against(dag)

        swapped = replace(
            stream,
            records=(stream.records[1], stream.records[0]),
            runtime_relocations=(
                replace(stream.runtime_relocations[1], record_index=0),
                replace(stream.runtime_relocations[0], record_index=1),
            ),
            address_relocations=tuple(replace(item, record_index=1) for item in stream.address_relocations),
        )
        with self.assertRaisesRegex(SchemaError, "DTE_ISSUE then DTE_WAIT"):
            _recreate(fragment, core_streams=(swapped,)).validate_against(dag)

        other_token = RuntimeSymbol("copy_other_token", RuntimeSymbolKind.DTE_TOKEN, copy.id)
        wait = stream.records[1]
        changed_wait = replace(
            wait,
            operands=(RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, other_token.id),),
        )
        wrong_token_stream = replace(
            stream,
            records=(stream.records[0], changed_wait),
            runtime_relocations=(stream.runtime_relocations[0], RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, other_token.id)),
        )
        with self.assertRaisesRegex(SchemaError, "shared issue/wait token"):
            _recreate(
                fragment,
                core_streams=(wrong_token_stream,),
                runtime_symbols=tuple(sorted(fragment.runtime_symbols + (other_token,), key=lambda item: item.id)),
            ).validate_against(dag)

        issue_operands = list(stream.records[0].operands)
        issue_operands[0] = RecordOperand.literal("direction", 1)
        bad_direction = replace(stream, records=(replace(stream.records[0], operands=tuple(issue_operands)), stream.records[1]))
        with self.assertRaisesRegex(SchemaError, "SPM_TO_SPM"):
            _recreate(fragment, core_streams=(bad_direction,)).validate_against(dag)
        issue_operands = list(stream.records[0].operands)
        payload_index = next(
            index
            for index, operand in enumerate(issue_operands)
            if operand.name == "payload_bits"
        )
        issue_operands[payload_index] = replace(
            issue_operands[payload_index], literal_value=8
        )
        bad_payload = replace(
            stream,
            records=(
                replace(stream.records[0], operands=tuple(issue_operands)),
                stream.records[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "SPM_TO_SPM"):
            _recreate(fragment, core_streams=(bad_payload,)).validate_against(dag)

        source = ProgramSymbol("reduce_source", ProgramSymbolKind.ABSOLUTE_ADDRESS, copy.id)
        destination = ProgramSymbol("reduce_destination", ProgramSymbolKind.ABSOLUTE_ADDRESS, copy.id)
        fake_reduce = RelocatableRecord(copy.id, RecordOpcode.LOCAL_REDUCE, (
            RecordOperand.literal("input_dtype", 0), RecordOperand.literal("accumulator_dtype", 1),
            RecordOperand.literal("output_dtype", 0), RecordOperand.literal("reduce_op", 1),
            RecordOperand.literal("rounding", 0), RecordOperand.literal("order", 0),
            RecordOperand.literal("input_count", 1), RecordOperand.literal("element_count", 16),
            RecordOperand.literal("input_stride_bytes", 32),
            RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS, source.id),
            RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS, destination.id),
        ))
        fake_stream = CoreFragmentStream(
            copy.logical_core, (fake_reduce,), (),
            (
                AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, source.id, 0),
                AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS, ProgramSymbolKind.ABSOLUTE_ADDRESS, destination.id, 0),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "DTE_ISSUE then DTE_WAIT"):
            _recreate(
                fragment,
                core_streams=(fake_stream,),
                runtime_symbols=(),
                program_symbols=tuple(sorted((source, destination), key=lambda item: item.id)),
            ).validate_against(dag)

    def test_asymmetric_two_rank_stream_round_trip_and_core_indexed_relocations(self) -> None:
        _ir1, _projection, _schedule_set, dag, fragment = _two_die_fragment()
        decoded = loads_dataclass(CommandFragment, canonical_json(fragment))
        self.assertEqual(decoded, fragment)
        self.assertEqual(canonical_digest(decoded), canonical_digest(fragment))
        self.assertEqual(len(fragment.core_streams), 2)
        left, right = fragment.core_streams
        self.assertEqual(left.runtime_relocations[0].record_index, 0)
        self.assertEqual(right.runtime_relocations[0].record_index, 0)
        self.assertNotEqual(left.records[0].opcode, right.records[0].opcode)
        self.assertNotEqual(left.runtime_relocations, right.runtime_relocations)
        recv_record = right.records[0]
        self.assertEqual(recv_record.operands[1].literal_value, 1)
        self.assertEqual(recv_record.operands[5].literal_value, 0)
        send_record = left.records[0]
        expected_sources_index = next(
            index
            for index, operand in enumerate(send_record.operands)
            if operand.name == "expected_sources"
        )
        operands = list(send_record.operands)
        operands[expected_sources_index] = replace(
            operands[expected_sources_index], literal_value=1
        )
        with self.assertRaisesRegex(SchemaError, "canonical zero literals"):
            _recreate(
                fragment,
                core_streams=(
                    replace(
                        left,
                        records=(replace(send_record, operands=tuple(operands)),),
                    ),
                    right,
                ),
            ).validate_against(dag)

    def test_fused_recv_wait_requires_async_shared_dte_token(self) -> None:
        ir1, projection, schedule_set = _with_recv_wait()
        dag = _create_global(ir1, projection, schedule_set)
        fragment = _fragment(dag)
        fragment.validate_against(dag)

        records = {
            record.source_global_action_id: (stream_index, record_index, record)
            for stream_index, stream in enumerate(fragment.core_streams)
            for record_index, record in enumerate(stream.records)
        }
        recv = next(
            action
            for action in dag.actions
            if action.task_kind is SemanticTaskKind.RECV
        )
        wait = next(
            action
            for action in dag.actions
            if action.task_kind is SemanticTaskKind.WAIT
        )
        send = next(
            action
            for action in dag.actions
            if action.task_kind is SemanticTaskKind.SEND
        )
        recv_stream_index, recv_record_index, recv_record = records[recv.id]
        wait_stream_index, wait_record_index, wait_record = records[wait.id]
        _send_stream_index, _send_record_index, send_record = records[send.id]
        self.assertEqual(recv_stream_index, wait_stream_index)
        self.assertLess(recv_record_index, wait_record_index)
        self.assertEqual(recv_record.operands[1].literal_value, 0)
        self.assertIs(recv_record.operands[5].kind, OperandKind.RUNTIME_SYMBOL)
        self.assertIs(wait_record.opcode, RecordOpcode.DTE_WAIT)
        self.assertEqual(
            recv_record.operands[5].symbol_ref,
            wait_record.operands[0].symbol_ref,
        )
        self.assertEqual(send_record.operands[2].literal_value, 1)
        self.assertEqual(send_record.operands[6].literal_value, 0)
        token = next(
            symbol
            for symbol in fragment.runtime_symbols
            if symbol.id == recv_record.operands[5].symbol_ref
        )
        self.assertIs(token.kind, RuntimeSymbolKind.DTE_TOKEN)
        self.assertEqual(token.source_ref, recv.runtime_binding.token_symbol)
        self.assertEqual(token.source_ref, wait.runtime_binding.token_symbol)

        recv_stream = fragment.core_streams[recv_stream_index]
        sync_operands = list(recv_record.operands)
        sync_operands[1] = RecordOperand.literal("completion", 1)
        sync_recv = replace(recv_record, operands=tuple(sync_operands))
        with self.assertRaisesRegex(SchemaError, "waited fused RECV"):
            _recreate(
                fragment,
                core_streams=tuple(
                    replace(
                        stream,
                        records=tuple(
                            sync_recv if index == recv_record_index else record
                            for index, record in enumerate(stream.records)
                        ),
                    )
                    if index == recv_stream_index
                    else stream
                    for index, stream in enumerate(fragment.core_streams)
                ),
            ).validate_against(dag)

        forged_token = RuntimeSymbol(
            "forged_wait_token",
            RuntimeSymbolKind.DTE_TOKEN,
            wait.runtime_binding.token_symbol,
        )
        wait_stream = fragment.core_streams[wait_stream_index]
        forged_wait = replace(
            wait_record,
            operands=(
                RecordOperand.runtime(
                    "token", RuntimeOperandField.DTE_TOKEN, forged_token.id
                ),
            ),
        )
        forged_relocations = tuple(
            replace(relocation, symbol_ref=forged_token.id)
            if relocation.record_index == wait_record_index
            and relocation.field is RuntimeOperandField.DTE_TOKEN
            else relocation
            for relocation in wait_stream.runtime_relocations
        )
        with self.assertRaisesRegex(SchemaError, "shared DTE token|sharing.*runtime token"):
            _recreate(
                fragment,
                core_streams=tuple(
                    replace(
                        stream,
                        records=tuple(
                            forged_wait if index == wait_record_index else record
                            for index, record in enumerate(stream.records)
                        ),
                        runtime_relocations=forged_relocations,
                    )
                    if index == wait_stream_index
                    else stream
                    for index, stream in enumerate(fragment.core_streams)
                ),
                runtime_symbols=tuple(
                    sorted(
                        fragment.runtime_symbols + (forged_token,),
                        key=lambda symbol: symbol.id,
                    )
                ),
            ).validate_against(dag)

        bad_wait = replace(wait, deps=())
        bad_dag = GlobalActionDAG.create(
            producer_pass=dag.producer_pass,
            **{
                **dag._semantic_key(),
                "actions": tuple(
                    bad_wait if action.id == wait.id else action
                    for action in dag.actions
                ),
            },
        )
        with self.assertRaisesRegex(SchemaError, "exactly one same-core RECV"):
            _recreate(
                fragment, source_global_dag_id=bad_dag.id
            ).validate_against(bad_dag)

    def test_origin_claim_and_contiguity_are_exact(self) -> None:
        _ir1, _projection, _schedule_set, dag, fragment = _two_die_fragment()
        send = next(
            action.id for action in dag.actions
            if action.task_kind is SemanticTaskKind.SEND
        )
        recv = next(
            action.id for action in dag.actions
            if action.task_kind is SemanticTaskKind.RECV
        )
        with self.assertRaisesRegex(SchemaError, "origin is not claimed"):
            _recreate(fragment, claimed_action_ids=(send,)).validate()
        send_only = _recreate(fragment, claimed_action_ids=(send,), core_streams=(fragment.core_streams[0],), runtime_symbols=tuple(symbol for symbol in fragment.runtime_symbols if symbol.source_ref == send), program_symbols=tuple(symbol for symbol in fragment.program_symbols if symbol.source_ref == send))
        with self.assertRaisesRegex(SchemaError, "at least one record"):
            _recreate(send_only, claimed_action_ids=fragment.claimed_action_ids).validate()

        a0, symbols0, _program0, relocs0, _address0 = _wait_record("a", "a")
        b, symbols1, _program1, relocs1, _address1 = _wait_record("b", "b")
        a2, _symbols2, _program2, relocs2, _address2 = _wait_record("a", "a")
        stream = CoreFragmentStream(
            LogicalCoreRef(0, 0), (a0, b, a2),
            (relocs0[0], replace(relocs1[0], record_index=1), replace(relocs2[0], record_index=2)), (),
        )
        noncontiguous = CommandFragment.create(
            producer_pass="fixture", source_global_dag_id="dag", kind=FragmentKind.ISA_REGION,
            claimed_action_ids=("a", "b"), core_streams=(stream,),
            runtime_symbols=tuple(sorted(symbols0 + symbols1, key=lambda item: item.id)),
            program_symbols=(), buffer_abi=(),
        )
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            noncontiguous.validate()

    def test_global_core_order_kind_and_transit_fail_closed(self) -> None:
        ir1, projection, schedule_set = _two_by_two_case()
        dag = _create_global(ir1, projection, schedule_set)
        transit = next(action for action in dag.actions if action.task_kind is SemanticTaskKind.TRANSIT)
        record, symbols, _program, relocations, _address = _wait_record(transit.id, "transit")
        fake = CommandFragment.create(
            producer_pass="fixture", source_global_dag_id=dag.id, kind=FragmentKind.ISA_REGION,
            claimed_action_ids=(transit.id,),
            core_streams=(CoreFragmentStream(LogicalCoreRef(1, 0), (record,), relocations, ()),),
            runtime_symbols=symbols, program_symbols=(), buffer_abi=(),
        )
        with self.assertRaisesRegex(SchemaError, "TRANSIT"):
            fake.validate_against(dag)

    def test_missing_double_and_literal_relocations_fail(self) -> None:
        _ir1, _projection, _schedule_set, _dag, fragment = _two_die_fragment()
        stream = fragment.core_streams[0]
        with self.assertRaisesRegex(SchemaError, "exactly one relocation"):
            _recreate(fragment, core_streams=(replace(stream, runtime_relocations=stream.runtime_relocations[1:]), fragment.core_streams[1])).validate()
        with self.assertRaisesRegex(SchemaError, "duplicate runtime relocation"):
            _recreate(fragment, core_streams=(replace(stream, runtime_relocations=(stream.runtime_relocations[0], stream.runtime_relocations[0]) + stream.runtime_relocations[1:]), fragment.core_streams[1])).validate()
        group = RuntimeSymbol("group_literal", RuntimeSymbolKind.GROUP, "literal")
        relocs = tuple(sorted(stream.runtime_relocations + (RuntimeRelocation(0, RuntimeOperandField.GROUP_ID, group.id),), key=lambda item: (item.record_index, list(RuntimeOperandField).index(item.field))))
        with self.assertRaisesRegex(SchemaError, "literals take no relocation"):
            _recreate(fragment, core_streams=(replace(stream, runtime_relocations=relocs), fragment.core_streams[1]), runtime_symbols=tuple(sorted(fragment.runtime_symbols + (group,), key=lambda item: item.id))).validate()

    def test_operand_symbol_kind_opcode_and_per_core_symbol_mismatch_fail(self) -> None:
        _ir1, _projection, _schedule_set, _dag, fragment = _two_die_fragment()
        send_stream, recv_stream = fragment.core_streams
        send_record = send_stream.records[0]
        source_index = next(index for index, operand in enumerate(send_record.operands) if operand.name == "source_address")
        operands = list(send_record.operands)
        operands[source_index] = replace(operands[source_index], operand_id=SemanticOperandId.DESTINATION_ADDRESS)
        with self.assertRaisesRegex(SchemaError, "semantic operand id"):
            replace(send_record, operands=tuple(operands)).validate("record")
        with self.assertRaisesRegex(SchemaError, "canonical ABI order"):
            replace(send_record, opcode=RecordOpcode.DTE_RECV).validate("record")

        bad_symbol = replace(fragment.runtime_symbols[0], kind=RuntimeSymbolKind.DTE_TOKEN)
        with self.assertRaisesRegex(SchemaError, "wrong-kind"):
            _recreate(fragment, runtime_symbols=(bad_symbol,) + fragment.runtime_symbols[1:]).validate()

        token = RuntimeSymbol("forged_async_token", RuntimeSymbolKind.DTE_TOKEN, "forged")
        token_index = next(index for index, operand in enumerate(recv_stream.records[0].operands) if operand.name == "token")
        token_record = replace(
            recv_stream.records[0],
            operands=tuple(
                RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id)
                if index == token_index
                else operand
                for index, operand in enumerate(recv_stream.records[0].operands)
            ),
        )
        token_relocations = tuple(sorted(
            recv_stream.runtime_relocations + (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),),
            key=lambda item: (item.record_index, list(RuntimeOperandField).index(item.field)),
        ))
        broken = _recreate(
            fragment,
            core_streams=(send_stream, replace(recv_stream, records=(token_record,), runtime_relocations=token_relocations)),
            runtime_symbols=tuple(sorted(fragment.runtime_symbols + (token,), key=lambda item: item.id)),
        )
        broken.validate()
        with self.assertRaisesRegex(SchemaError, "token=0"):
            broken.validate_against(_dag)

    def test_program_symbol_numeric_abi_and_address_kind_are_strict(self) -> None:
        self.assertEqual(int(ProgramSymbolKind.ABSOLUTE_ADDRESS), 1)
        self.assertEqual(int(ProgramSymbolKind.SRAM_REGION), 2)
        self.assertEqual(int(ProgramSymbolKind.SRAM_LABEL), 3)
        self.assertEqual(int(SemanticOperandId.SOURCE_ADDRESS), 4)
        _ir1, _projection, _schedule_set, _dag, fragment = _two_die_fragment()
        stream = fragment.core_streams[0]
        target_symbol_id = stream.address_relocations[0].symbol_ref
        symbol = next(
            item for item in fragment.program_symbols
            if item.id == target_symbol_id
        )
        bad_symbol = replace(symbol, kind=ProgramSymbolKind.SRAM_LABEL)
        bad_address = replace(stream.address_relocations[0], symbol_kind=ProgramSymbolKind.SRAM_LABEL)
        with self.assertRaisesRegex(SchemaError, "illegal for opcode/operand"):
            _recreate(
                fragment,
                program_symbols=tuple(
                    bad_symbol if item.id == target_symbol_id else item
                    for item in fragment.program_symbols
                ),
                core_streams=(
                    replace(stream, address_relocations=(bad_address,)),
                    fragment.core_streams[1],
                ),
            ).validate()

    def test_region_manifest_is_one_region_plan_and_derives_target_dies(self) -> None:
        _ir1, _projection, _schedule_set, dag, fragment = _two_die_fragment()
        send = next(action for action in dag.actions if action.task_kind is SemanticTaskKind.SEND)
        fragment = _fragment(dag, (send,))
        region = RegionManifest.create(
            producer_pass="region_fixture", region_id=send.region_id,
            fusion_plan_id=send.origin_ref.plan_id, target_dies=(0,), fragment=fragment,
        )
        region.validate_against(dag)
        decoded = loads_dataclass(RegionManifest, canonical_json(region))
        self.assertEqual(decoded, region)
        with self.assertRaisesRegex(SchemaError, "derive"):
            replace(region, target_dies=(1,)).validate()
        with self.assertRaisesRegex(SchemaError, "fusion plan"):
            RegionManifest.create(
                producer_pass=region.producer_pass,
                region_id=region.region_id,
                fusion_plan_id="wrong",
                target_dies=region.target_dies,
                fragment=region.fragment,
            ).validate_against(dag)

    def test_buffer_abi_uses_schedule_binding_and_named_region_refs(self) -> None:
        _ir1, _projection, schedule_set, dag, fragment = _two_die_fragment()
        action = dag.actions[0]
        schedule = schedule_set.schedules[0]
        binding = schedule.buffer_bindings[0]
        abi = BufferABI(
            id="abi_send", schedule_id=schedule.id, binding_id=binding.id,
            value_id=binding.value_id, logical_core=action.logical_core,
            tensor_slice=binding.tensor_slice, region_ref=binding.region_ref,
            region_offset_bytes=binding.region_offset_bytes, size_bytes=binding.size_bytes,
            alignment_bytes=binding.alignment_bytes, banks=binding.banks,
            storage_id=binding.storage_id, alias_of=binding.alias_of,
            lifetime_start=binding.lifetime_start,
            lifetime_end_exclusive=binding.lifetime_end_exclusive,
            dtype=binding.dtype, layout=binding.layout, ownership=binding.ownership,
        )
        _recreate(fragment, buffer_abi=(abi,)).validate()

    def test_plan_barrier_command_is_canonical_for_tp2_and_tp4(self) -> None:
        for tp in (2, 4):
            with self.subTest(tp=tp):
                dag, participants, fragment = _plan_barrier_fragment(tp)
                fragment.validate_against(dag)
                counts = tuple(
                    len(stream.records) for stream in fragment.core_streams
                )
                self.assertEqual(counts, (2 * (tp - 1), *((2,) * (tp - 1))))
                self.assertEqual(sum(counts), 4 * (tp - 1))
                self.assertEqual(
                    sum(
                        symbol.kind is RuntimeSymbolKind.EVENT_TAG
                        for symbol in fragment.runtime_symbols
                    ),
                    2 * (tp - 1),
                )
                self.assertEqual(
                    canonical_plan_barrier_core_symbol(dag.id, participants[0]),
                    canonical_plan_barrier_core_symbol(dag.id, participants[0]),
                )

    def test_plan_barrier_command_rejects_missing_rank_and_reordered_arrivals(self) -> None:
        dag, participants, fragment = _plan_barrier_fragment(4)
        missing = participants[-1]
        broken = _recreate(
            fragment,
            claimed_action_ids=tuple(
                action_id
                for action_id in fragment.claimed_action_ids
                if action_id != missing.id
            ),
            core_streams=tuple(
                stream
                for stream in fragment.core_streams
                if stream.logical_core != missing.logical_core
            ),
        )
        broken.validate()
        with self.assertRaisesRegex(SchemaError, "all participant actions"):
            broken.validate_against(dag)

        leader = participants[0]
        leader_index = next(
            index
            for index, stream in enumerate(fragment.core_streams)
            if stream.logical_core == leader.logical_core
        )
        leader_stream = fragment.core_streams[leader_index]
        records = list(leader_stream.records)
        records[0], records[1] = records[1], records[0]
        changed_stream = replace(
            leader_stream,
            records=tuple(records),
            runtime_relocations=_event_relocations(tuple(records)),
        )
        reordered = _recreate(
            fragment,
            core_streams=tuple(
                changed_stream if index == leader_index else stream
                for index, stream in enumerate(fragment.core_streams)
            ),
        )
        reordered.validate()
        with self.assertRaisesRegex(SchemaError, "participant-ordered"):
            reordered.validate_against(dag)

    def test_plan_barrier_command_rejects_forged_symbol_and_group_scope(self) -> None:
        dag, participants, fragment = _plan_barrier_fragment(2)
        event = next(
            symbol
            for symbol in fragment.runtime_symbols
            if symbol.kind is RuntimeSymbolKind.EVENT_TAG
        )
        forged = _recreate(
            fragment,
            runtime_symbols=tuple(
                replace(symbol, source_ref="forged_barrier")
                if symbol.id == event.id
                else symbol
                for symbol in fragment.runtime_symbols
            ),
        )
        forged.validate()
        with self.assertRaisesRegex(SchemaError, "runtime symbols"):
            forged.validate_against(dag)

        group_action = replace(
            participants[0],
            sync=replace(
                participants[0].sync,
                barrier=replace(
                    participants[0].sync.barrier,
                    scope=BarrierScope.GROUP,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "standalone PLAN barrier"):
            canonical_plan_barrier_core_symbol(dag.id, group_action)


if __name__ == "__main__":
    unittest.main()
