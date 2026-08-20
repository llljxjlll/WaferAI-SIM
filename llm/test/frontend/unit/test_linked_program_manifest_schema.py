from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.lowering import LoweringContext
from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import ComputeOperand
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    AddressOperandBinding,
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
    PlanBarrierEventPhase,
    canonical_plan_barrier_core_symbol,
    canonical_plan_barrier_event_symbol,
    _validate_address_operand_closure,
    _validate_fused_recv_wait_token_closure,
    _validate_local_reduce_absolute_alignment,
    _validate_local_reduce_containment_witness,
    _validate_runtime_binding_ref,
)
from llm.frontend.wafer_frontend.schema.global_action import (
    GlobalActionDAG,
    LogicalCoreRef,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferAccess,
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    IntraDieDAG,
    IntraDieSchedule,
    IntraDieScheduleSet,
    IntraDieValue,
    IR2ProjectionResult,
    TaskBufferUse,
    TensorSlice,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)

from test_artifact_manifest_schema import (
    _compute_fragment,
    _fragment,
    _plan_barrier_fragment,
    _recreate,
)
from test_global_action_schema import _create_global, _two_die_case, _with_recv_wait
from test_lowering_context import valid_lowering_context
from test_ir2_schema import valid_reduce_case
from test_naive_intra_die import _complete_projection
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, GemmPartition, GemmWorkload, OpKind


def _abi(schedule, binding, core: LogicalCoreRef) -> BufferABI:
    return BufferABI(
        id=f"abi_{binding.id}",
        schedule_id=schedule.id,
        binding_id=binding.id,
        value_id=binding.value_id,
        logical_core=core,
        tensor_slice=binding.tensor_slice,
        region_ref=binding.region_ref,
        region_offset_bytes=binding.region_offset_bytes,
        size_bytes=binding.size_bytes,
        alignment_bytes=binding.alignment_bytes,
        banks=binding.banks,
        storage_id=binding.storage_id,
        alias_of=binding.alias_of,
        lifetime_start=binding.lifetime_start,
        lifetime_end_exclusive=binding.lifetime_end_exclusive,
        dtype=binding.dtype,
        layout=binding.layout,
        ownership=binding.ownership,
    )


def _two_input_lowering_context() -> LoweringContext:
    """Upgrade the ordinary fixture to model GEMM activation and weight separately."""

    base = valid_lowering_context()
    old_node = base.ir1.nodes[0]
    input_value, output_value = base.ir1.values
    weight_value = replace(
        input_value,
        id="p_v_weight",
        shape=(256, 128),
        logical_layout="KN",
        consumers=(old_node.id,),
    )
    gemm = GemmWorkload(
        (32, 128, 256),
        (32, 128, 256),
        GemmPartition.REPLICATED,
        input_value.dtype,
    )
    node = replace(
        old_node, kind=OpKind.GEMM, inputs=(input_value.id, weight_value.id),
        workload=gemm, impl_ref="matmul_forward",
    )
    ir1_fields = base.ir1._semantic_key()
    ir1_fields.update(
        nodes=(node,),
        values=(input_value, weight_value, output_value),
    )
    ir1 = IR1.create(producer_pass=base.ir1.producer_pass, **ir1_fields)

    dags = []
    for old_dag in base.projection.dags:
        old_task = old_dag.tasks[0]
        compute = replace(
            old_task.compute,
            op_kind=OpKind.GEMM,
            workload=gemm,
            impl_ref="matmul_forward",
            inputs=(
                ComputeOperand(input_value.id, "lhs"),
                ComputeOperand(weight_value.id, "rhs"),
            ),
            outputs=(ComputeOperand(output_value.id, "output"),),
        )
        task = replace(
            old_task,
            op_kind=OpKind.GEMM,
            read_values=(input_value.id, weight_value.id),
            compute=compute,
        )
        local_input, local_output = old_dag.values
        local_weight = IntraDieValue(
            weight_value.id,
            weight_value.id,
            weight_value.shape,
            weight_value.dtype,
            weight_value.logical_layout,
            weight_value.sharding,
            weight_value.alias_set,
            (),
            (task.id,),
        )
        dag_fields = old_dag._semantic_key()
        dag_fields.update(
            source_ir1_id=ir1.id,
            tasks=(task,),
            values=(local_input, local_weight, local_output),
        )
        dags.append(
            IntraDieDAG.create(producer_pass=old_dag.producer_pass, **dag_fields)
        )
    projection_fields = base.projection._semantic_key()
    projection_fields.update(source_ir1_id=ir1.id, dags=tuple(dags))
    projection = IR2ProjectionResult.create(
        producer_pass=base.projection.producer_pass, **projection_fields
    )

    schedules = []
    for old_schedule, dag in zip(base.schedule_set.schedules, dags):
        input_binding, old_output_binding = old_schedule.buffer_bindings
        weight_offset = input_binding.region_offset_bytes + input_binding.size_bytes
        weight_bytes = 256 * 128 * 2
        output_offset = weight_offset + weight_bytes
        weight_binding = BufferBinding(
            f"binding_weight_d{old_schedule.die_id}",
            weight_value.id,
            TensorSlice(weight_value.id, (0, 0), weight_value.shape),
            input_binding.core_id,
            input_binding.region_ref,
            weight_offset,
            weight_bytes,
            input_binding.alignment_bytes,
            input_binding.banks,
            f"storage_weight_d{old_schedule.die_id}",
            None,
            BufferOwnership.BORROWED,
            0,
            1,
            weight_value.dtype,
            weight_value.logical_layout,
        )
        output_binding = replace(
            old_output_binding, region_offset_bytes=output_offset
        )
        old_input_use, old_output_use = old_schedule.task_buffer_uses
        weight_use = TaskBufferUse(
            dag.tasks[0].id,
            weight_binding.id,
            BufferAccess.READ,
            BufferUseRole.COMP_INPUT,
            1,
            None,
            weight_binding.tensor_slice,
        )
        schedule_fields = old_schedule._semantic_key()
        schedule_fields.update(
            dag_id=dag.id,
            buffer_bindings=(input_binding, weight_binding, output_binding),
            task_buffer_uses=(old_input_use, weight_use, old_output_use),
        )
        schedules.append(
            IntraDieSchedule.create(
                producer_pass=old_schedule.producer_pass, **schedule_fields
            )
        )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass=base.schedule_set.producer_pass,
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=tuple(schedules),
    )
    global_dag = _create_global(ir1, projection, schedule_set)
    context = LoweringContext(
        ir1,
        (),
        (),
        projection,
        schedule_set,
        global_dag,
    )
    context.validate()
    return context


def valid_linked_manifest(*, with_recv_wait: bool = False):
    ir1, projection, schedule_set = (
        _with_recv_wait() if with_recv_wait else _two_die_case()
    )
    dag = _create_global(ir1, projection, schedule_set)
    fragment = _fragment(dag)

    region_symbol = ProgramSymbol(
        "program_region_sram_main", ProgramSymbolKind.SRAM_REGION, "sram_main"
    )
    streams = []
    for stream in fragment.core_streams:
        records = []
        for record in stream.records:
            records.append(
                replace(
                    record,
                    operands=tuple(
                        replace(operand, symbol_ref=region_symbol.id)
                        if operand.kind.value == "address_symbol"
                        else operand
                        for operand in record.operands
                    ),
                )
            )
        streams.append(
            replace(
                stream,
                records=tuple(records),
                address_relocations=tuple(
                    replace(relocation, symbol_ref=region_symbol.id)
                    for relocation in stream.address_relocations
                ),
            )
        )

    schedules = {schedule.die_id: schedule for schedule in schedule_set.schedules}
    actions_by_core = {
        core: tuple(
            sorted(
                (
                    action
                    for action in dag.actions
                    if action.logical_core == core
                ),
                key=lambda action: action.core_order_index,
            )
        )
        for core in {
            action.logical_core for action in dag.actions if action.logical_core is not None
        }
    }
    abis = []
    for core, core_actions in sorted(
        actions_by_core.items(),
        key=lambda item: (item[0].die_id, item[0].local_core_id),
    ):
        schedule = schedules[core.die_id]
        for action in core_actions:
            for use in action.buffer_uses:
                binding = next(
                    binding
                    for binding in schedule.buffer_bindings
                    if binding.id == use.binding_id
                )
                abi = _abi(schedule, binding, core)
                if abi not in abis:
                    abis.append(abi)
    abis.sort(key=lambda item: item.id)
    fragment = _recreate(
        fragment,
        core_streams=tuple(streams),
        program_symbols=(region_symbol,),
        buffer_abi=tuple(abis),
    )

    active = tuple(
        sorted(actions_by_core, key=lambda core: (core.die_id, core.local_core_id))
    )
    core_bindings = []
    for core in active:
        die = ir1.fabric.dies[core.die_id]
        spec = next(item for item in die.cores if item.local_core_id == core.local_core_id)
        core_bindings.append(
            CoreRuntimeBinding(core, spec.id, spec.runtime_core_id, spec.sram_profile_ref)
        )
    fragment_streams = {
        stream.logical_core: stream for stream in fragment.core_streams
    }
    core_streams = tuple(
        LinkedCoreStream(
            binding.logical_core,
            binding.runtime_core_id,
            tuple(
                LinkedRecordRef(
                    fragment.id, record_index, record.source_global_action_id
                )
                for record_index, record in enumerate(
                    fragment_streams[binding.logical_core].records
                )
            ),
        )
        for binding in core_bindings
    )

    runtime_definitions = []
    actions = {action.id: action for action in dag.actions}
    for symbol in fragment.runtime_symbols:
        token_actions = tuple(
            actions[record.source_global_action_id]
            for stream in fragment.core_streams
            for relocation in stream.runtime_relocations
            if relocation.symbol_ref == symbol.id
            for record in (stream.records[relocation.record_index],)
        )
        action = actions.get(symbol.source_ref)
        if symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            assert action is not None
            peer = next(
                candidate
                for candidate in dag.actions
                if candidate.flow_id == action.flow_id and candidate.id != action.id
            )
            cores = (peer.logical_core,)
            source = destination = None
        elif symbol.kind is RuntimeSymbolKind.DTE_TOKEN:
            recv = next(
                (
                    candidate
                    for candidate in token_actions
                    if candidate.task_kind.value == "recv"
                ),
                None,
            )
            wait = next(
                (
                    candidate
                    for candidate in token_actions
                    if candidate.task_kind.value == "wait"
                ),
                None,
            )
            owner = recv or token_actions[0]
            cores = (owner.logical_core,)
            source, destination = owner.id, wait.id if wait is not None else None
        else:
            assert action is not None
            peer = next(
                candidate
                for candidate in dag.actions
                if candidate.flow_id == action.flow_id and candidate.id != action.id
            )
            cores = tuple(
                sorted(
                    (action.logical_core, peer.logical_core),
                    key=lambda core: (core.die_id, core.local_core_id),
                )
            )
            source, destination = action.id, peer.id
        runtime_definitions.append(
            RuntimeSymbolDefinition(symbol, cores, source, destination)
        )

    start_events = []
    for index, core in enumerate(active):
        first_action = actions_by_core[core][0]
        symbol = RuntimeSymbol(
            f"start_{index}", RuntimeSymbolKind.START_TAG, first_action.id
        )
        runtime_definitions.append(RuntimeSymbolDefinition(symbol, (core,), None, None))
        start_events.append(LogicalStartEvent(core, symbol.id, 1))
    runtime_definitions.sort(key=lambda item: item.symbol.id)

    interface = FragmentInterface(
        fragment.id,
        (),
        tuple(symbol.id for symbol in fragment.runtime_symbols),
        (),
        (region_symbol.id,),
        (),
        (),
    )
    program_definition = ProgramSymbolDefinition(
        region_symbol, "sram", 0, 1 << 20, active
    )
    abi_by_core = {abi.logical_core: abi for abi in abis}
    address_bindings = tuple(
        AddressOperandBinding(
            fragment.id,
            stream.logical_core,
            relocation.record_index,
            relocation.operand_id,
            (abi_by_core[stream.logical_core].id,),
            (abi_by_core[stream.logical_core].tensor_slice,),
        )
        for stream in streams
        for relocation in stream.address_relocations
    )
    envelope = ProgramControlEnvelope(
        active,
        tuple(start_events),
        active,
        active,
        active,
        EmptyCoreAckPolicy.INCLUDE_EMPTY,
        ProgramFailurePolicy.ABORT_ALL,
    )
    inputs = tuple(
        sorted(
            (
                ManifestInputDigest(ManifestInputKind.IR1, ir1.id, ir1.schema_version, canonical_digest(ir1)),
                ManifestInputDigest(ManifestInputKind.IR2_PROJECTION, projection.id, projection.schema_version, canonical_digest(projection)),
                ManifestInputDigest(ManifestInputKind.SCHEDULE_SET, schedule_set.id, schedule_set.schema_version, canonical_digest(schedule_set)),
                ManifestInputDigest(ManifestInputKind.GLOBAL_ACTION_DAG, dag.id, dag.schema_version, canonical_digest(dag)),
                ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT, fragment.id, fragment.schema_version, canonical_digest(fragment)),
            ),
            key=lambda item: (item.kind.value, item.artifact_id),
        )
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="linked_fixture",
        capabilities=0,
        source_ir1_id=ir1.id,
        source_projection_id=projection.id,
        source_schedule_set_id=schedule_set.id,
        source_global_dag_id=dag.id,
        input_digests=inputs,
        fragments=(fragment,),
        fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings),
        core_streams=core_streams,
        runtime_symbol_definitions=tuple(runtime_definitions),
        program_symbol_definitions=(program_definition,),
        address_operand_bindings=address_bindings,
        core_groups=(),
        envelope=envelope,
    )
    return ir1, projection, schedule_set, dag, manifest


def _recreate_manifest(
    manifest: LinkedProgramManifest, **changes: object
) -> LinkedProgramManifest:
    fields = manifest._semantic_key()
    fields.update(changes)
    return LinkedProgramManifest.create(producer_pass=manifest.producer_pass, **fields)


def _exact_input_digests(context: LoweringContext, fragments: tuple) -> tuple:
    artifacts = [
        (ManifestInputKind.IR1, context.ir1),
        *((ManifestInputKind.FUSION_PLAN, plan) for plan in context.fusion_plans),
        *((ManifestInputKind.STANDALONE_PLAN, plan) for plan in context.standalone_plans),
        (ManifestInputKind.IR2_PROJECTION, context.projection),
        (ManifestInputKind.SCHEDULE_SET, context.schedule_set),
        (ManifestInputKind.GLOBAL_ACTION_DAG, context.global_dag),
        *((ManifestInputKind.COMMAND_FRAGMENT, fragment) for fragment in fragments),
    ]
    return tuple(
        sorted(
            (
                ManifestInputDigest(
                    kind,
                    artifact.id,
                    artifact.schema_version,
                    canonical_digest(artifact),
                )
                for kind, artifact in artifacts
            ),
            key=lambda item: (item.kind.value, item.artifact_id),
        )
    )


def valid_exact_linked_manifest() -> tuple[LoweringContext, LinkedProgramManifest]:
    context = _two_input_lowering_context()
    schedules = {schedule.id: schedule for schedule in context.schedule_set.schedules}
    fragments = []
    interfaces = []
    address_bindings = []
    program_definitions = []
    core_bindings = []
    runtime_definitions = []
    start_events = []

    for start_index, action in enumerate(context.global_dag.actions):
        schedule = schedules[action.source.schedule_id]
        bindings = {binding.id: binding for binding in schedule.buffer_bindings}
        uses = {
            (use.role, use.operand_index): use for use in action.buffer_uses
        }
        relocation_bindings = {
            SemanticOperandId.SRAM_BIND_INPUT_0: bindings[
                uses[(BufferUseRole.COMP_INPUT, 0)].binding_id
            ],
            SemanticOperandId.SRAM_BIND_OUTPUT: bindings[
                uses[(BufferUseRole.COMP_OUTPUT, 0)].binding_id
            ],
            SemanticOperandId.COMPUTE_INPUT_ADDRESS: bindings[
                uses[(BufferUseRole.COMP_INPUT, 0)].binding_id
            ],
            SemanticOperandId.COMPUTE_DATA_ADDRESS: bindings[
                uses[(BufferUseRole.COMP_INPUT, 1)].binding_id
            ],
            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS: bindings[
                uses[(BufferUseRole.COMP_OUTPUT, 0)].binding_id
            ],
        }
        abis = {
            operand_id: _abi(schedule, binding, action.logical_core)
            for operand_id, binding in relocation_bindings.items()
        }
        fragment = _compute_fragment(
            context.global_dag,
            action,
            RecordOpcode.MATMUL,
            (
                1,
                action.compute.workload.rank_shape[0],
                action.compute.workload.rank_shape[2],
                action.compute.workload.rank_shape[1],
            ),
        )
        fragment = _recreate(
            fragment,
            buffer_abi=tuple(sorted(set(abis.values()), key=lambda item: item.id)),
        )
        fragments.append(fragment)
        interfaces.append(
            FragmentInterface(
                fragment.id,
                (),
                (),
                (),
                tuple(symbol.id for symbol in fragment.program_symbols),
                (),
                (),
            )
        )
        for relocation in fragment.core_streams[0].address_relocations:
            abi = abis[relocation.operand_id]
            address_bindings.append(
                AddressOperandBinding(
                    fragment.id,
                    action.logical_core,
                    relocation.record_index,
                    relocation.operand_id,
                    (abi.id,),
                    (abi.tensor_slice,),
                )
            )
            symbol = next(
                item for item in fragment.program_symbols if item.id == relocation.symbol_ref
            )
            program_definitions.append(
                ProgramSymbolDefinition(
                    symbol,
                    symbol.id,
                    (
                        0
                        if symbol.kind is ProgramSymbolKind.SRAM_LABEL
                        else abi.region_offset_bytes
                    ),
                    (
                        0
                        if symbol.kind is ProgramSymbolKind.SRAM_LABEL
                        else abi.size_bytes
                    ),
                    (action.logical_core,),
                )
            )

        die = next(
            item
            for item in context.ir1.fabric.dies
            if item.id == action.logical_core.die_id
        )
        core = next(
            item
            for item in die.cores
            if item.local_core_id == action.logical_core.local_core_id
        )
        core_bindings.append(
            CoreRuntimeBinding(
                action.logical_core,
                core.id,
                core.runtime_core_id,
                core.sram_profile_ref,
            )
        )
        start_symbol = RuntimeSymbol(
            f"start_{start_index}", RuntimeSymbolKind.START_TAG, action.id
        )
        runtime_definitions.append(
            RuntimeSymbolDefinition(start_symbol, (action.logical_core,), None, None)
        )
        start_events.append(LogicalStartEvent(action.logical_core, start_symbol.id, 1))

    fragments = tuple(sorted(fragments, key=lambda item: item.id))
    fragment_by_action = {
        fragment.claimed_action_ids[0]: fragment for fragment in fragments
    }
    action_by_core = {
        action.logical_core: action for action in context.global_dag.actions
    }
    core_bindings = tuple(
        sorted(
            core_bindings,
            key=lambda item: (
                item.logical_core.die_id,
                item.logical_core.local_core_id,
            ),
        )
    )
    core_streams = tuple(
        LinkedCoreStream(
            binding.logical_core,
            binding.runtime_core_id,
            (
                LinkedRecordRef(
                    fragment_by_action[action_by_core[binding.logical_core].id].id,
                    0,
                    action_by_core[binding.logical_core].id,
                ),
                LinkedRecordRef(
                    fragment_by_action[action_by_core[binding.logical_core].id].id,
                    1,
                    action_by_core[binding.logical_core].id,
                ),
            ),
        )
        for binding in core_bindings
    )
    active = tuple(binding.logical_core for binding in core_bindings)
    envelope = ProgramControlEnvelope(
        active,
        tuple(
            sorted(
                start_events,
                key=lambda item: (
                    item.target_core.die_id,
                    item.target_core.local_core_id,
                    item.tag_symbol_ref,
                ),
            )
        ),
        active,
        active,
        active,
        EmptyCoreAckPolicy.INCLUDE_EMPTY,
        ProgramFailurePolicy.ABORT_ALL,
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="exact_linked_fixture",
        capabilities=0,
        source_ir1_id=context.ir1.id,
        source_projection_id=context.projection.id,
        source_schedule_set_id=context.schedule_set.id,
        source_global_dag_id=context.global_dag.id,
        input_digests=_exact_input_digests(context, fragments),
        fragments=fragments,
        fragment_interfaces=tuple(
            sorted(interfaces, key=lambda item: item.fragment_id)
        ),
        core_bindings=core_bindings,
        core_streams=core_streams,
        runtime_symbol_definitions=tuple(
            sorted(runtime_definitions, key=lambda item: item.symbol.id)
        ),
        program_symbol_definitions=tuple(
            sorted(program_definitions, key=lambda item: item.symbol.id)
        ),
        address_operand_bindings=tuple(
            sorted(
                address_bindings,
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                    item.fragment_id,
                    item.fragment_record_index,
                    int(item.operand_id),
                ),
            )
        ),
        core_groups=(),
        envelope=envelope,
    )
    return context, manifest


def _reduce_operand_case():
    ir1, dag, schedule = valid_reduce_case()
    projection = IR2ProjectionResult.create(
        producer_pass="reduce_operand_projection_fixture",
        source_ir1_id=ir1.id,
        fusion_plan_ids=("fp_reduce",),
        standalone_collective_plan_ids=(),
        dags=(dag,),
    )
    schedule_set = IntraDieScheduleSet.create(
        producer_pass="reduce_operand_schedule_fixture",
        source_projection_id=projection.id,
        source_ir1_id=ir1.id,
        schedules=(schedule,),
    )
    action = _create_global(ir1, projection, schedule_set).actions[0]
    core = action.logical_core
    assert core is not None
    rank0 = replace(_abi(schedule, schedule.buffer_bindings[0], core), id="z_rank0")
    rank1 = replace(_abi(schedule, schedule.buffer_bindings[1], core), id="a_rank1")
    output = replace(_abi(schedule, schedule.buffer_bindings[2], core), id="m_output")
    source = ProgramSymbol(
        "reduce_source", ProgramSymbolKind.ABSOLUTE_ADDRESS, action.id
    )
    destination = ProgramSymbol(
        "reduce_destination", ProgramSymbolKind.ABSOLUTE_ADDRESS, action.id
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
    by_binding = {
        (schedule.id, schedule.buffer_bindings[0].id): rank0,
        (schedule.id, schedule.buffer_bindings[1].id): rank1,
        (schedule.id, schedule.buffer_bindings[2].id): output,
    }
    by_id = {abi.id: abi for abi in (rank0, rank1, output)}
    closure = AddressOperandBinding(
        "reduce_fragment",
        core,
        0,
        SemanticOperandId.SOURCE_ADDRESS,
        (rank0.id, rank1.id),
        tuple(
            use.tensor_slice
            for rank in action.reduction.input_ranks
            for use in action.buffer_uses
            if use.role is BufferUseRole.REDUCE_INPUT
            and use.contribution_rank == rank
        ),
    )
    return action, record, closure, by_binding, by_id


def valid_cross_fragment_event_manifest() -> LinkedProgramManifest:
    source_core = LogicalCoreRef(0, 0)
    destination_core = LogicalCoreRef(1, 0)
    source_action = "event_source_action"
    destination_action = "event_destination_action"
    event = RuntimeSymbol("event_shared", RuntimeSymbolKind.EVENT_TAG, "event_dep")
    source = RuntimeSymbol(
        "event_source_core", RuntimeSymbolKind.RUNTIME_CORE, "event_dep"
    )
    destination = RuntimeSymbol(
        "event_destination_core", RuntimeSymbolKind.RUNTIME_CORE, "event_dep"
    )
    runtime_symbols = tuple(sorted((event, source, destination), key=lambda item: item.id))

    def event_fragment(
        *, action_id: str, core: LogicalCoreRef, opcode: RecordOpcode
    ) -> CommandFragment:
        operands = (
            RecordOperand.runtime(
                "source_core", RuntimeOperandField.SOURCE_CORE, source.id
            ),
            RecordOperand.runtime(
                "destination_core",
                RuntimeOperandField.DESTINATION_CORE,
                destination.id,
            ),
            RecordOperand.runtime("tag", RuntimeOperandField.EVENT_TAG, event.id),
        )
        if opcode is RecordOpcode.EVENT_WAIT:
            operands = (*operands, RecordOperand.literal("count", 1))
        record = RelocatableRecord(action_id, opcode, operands)
        relocations = tuple(
            sorted(
                (
                    RuntimeRelocation(0, RuntimeOperandField.EVENT_TAG, event.id),
                    RuntimeRelocation(0, RuntimeOperandField.SOURCE_CORE, source.id),
                    RuntimeRelocation(
                        0, RuntimeOperandField.DESTINATION_CORE, destination.id
                    ),
                ),
                key=lambda item: (
                    item.record_index,
                    list(RuntimeOperandField).index(item.field),
                ),
            )
        )
        return CommandFragment.create(
            producer_pass="event_fragment_fixture",
            source_global_dag_id="event_dag",
            kind=FragmentKind.ISA_REGION,
            claimed_action_ids=(action_id,),
            core_streams=(CoreFragmentStream(core, (record,), relocations, ()),),
            runtime_symbols=runtime_symbols,
            program_symbols=(),
            buffer_abi=(),
        )

    source_fragment = event_fragment(
        action_id=source_action,
        core=source_core,
        opcode=RecordOpcode.EVENT_SET,
    )
    destination_fragment = event_fragment(
        action_id=destination_action,
        core=destination_core,
        opcode=RecordOpcode.EVENT_WAIT,
    )
    fragments = tuple(sorted((source_fragment, destination_fragment), key=lambda item: item.id))
    interfaces = tuple(
        sorted(
            (
                FragmentInterface(
                    source_fragment.id,
                    (),
                    tuple(symbol.id for symbol in runtime_symbols),
                    (),
                    (),
                    (),
                    (EventCredit(event.id, 1),),
                ),
                FragmentInterface(
                    destination_fragment.id,
                    tuple(symbol.id for symbol in runtime_symbols),
                    (),
                    (),
                    (),
                    (EventCredit(event.id, 1),),
                    (),
                ),
            ),
            key=lambda item: item.fragment_id,
        )
    )
    active = (source_core, destination_core)
    core_bindings = (
        CoreRuntimeBinding(source_core, "core_0_0", 0, "sram_default"),
        CoreRuntimeBinding(destination_core, "core_1_0", 16, "sram_default"),
    )
    fragment_by_action = {
        fragment.claimed_action_ids[0]: fragment for fragment in fragments
    }
    core_streams = (
        LinkedCoreStream(
            source_core,
            0,
            (LinkedRecordRef(fragment_by_action[source_action].id, 0, source_action),),
        ),
        LinkedCoreStream(
            destination_core,
            16,
            (
                LinkedRecordRef(
                    fragment_by_action[destination_action].id,
                    0,
                    destination_action,
                ),
            ),
        ),
    )
    starts = (
        RuntimeSymbol("event_start_0", RuntimeSymbolKind.START_TAG, source_action),
        RuntimeSymbol(
            "event_start_1", RuntimeSymbolKind.START_TAG, destination_action
        ),
    )
    definitions = (
        RuntimeSymbolDefinition(destination, (destination_core,), None, None),
        RuntimeSymbolDefinition(event, active, source_action, destination_action),
        RuntimeSymbolDefinition(source, (source_core,), None, None),
        RuntimeSymbolDefinition(starts[0], (source_core,), None, None),
        RuntimeSymbolDefinition(starts[1], (destination_core,), None, None),
    )
    envelope = ProgramControlEnvelope(
        active,
        (
            LogicalStartEvent(source_core, starts[0].id, 1),
            LogicalStartEvent(destination_core, starts[1].id, 1),
        ),
        active,
        active,
        active,
        EmptyCoreAckPolicy.INCLUDE_EMPTY,
        ProgramFailurePolicy.ABORT_ALL,
    )
    return LinkedProgramManifest.create(
        producer_pass="event_linked_fixture",
        capabilities=0,
        source_ir1_id="event_ir1",
        source_projection_id="event_projection",
        source_schedule_set_id="event_schedule_set",
        source_global_dag_id="event_dag",
        input_digests=(
            ManifestInputDigest(ManifestInputKind.IR1, "event_ir1", "v1", "0" * 64),
        ),
        fragments=fragments,
        fragment_interfaces=interfaces,
        core_bindings=core_bindings,
        core_streams=core_streams,
        runtime_symbol_definitions=definitions,
        program_symbol_definitions=(),
        address_operand_bindings=(),
        core_groups=(),
        envelope=envelope,
    )


def _plan_barrier_linked_fixture(tp: int):
    ir1, projection = _complete_projection(tp=tp, large_sram=True)
    schedule_set = NaiveIntraDiePolicy().schedule(projection, ir1)
    dag, participants, fragment = _plan_barrier_fragment(
        tp, barrier_only=True
    )
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(ir1, skeleton, ir1.profile)
        for skeleton in ir1.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in ir1.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(ir1, node, ir1.profile)
        for node in ir1.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and node.workload.collective is CollectiveKind.ALL_GATHER
    )

    core_symbol_actions = {
        canonical_plan_barrier_core_symbol(dag.id, action).id: action
        for action in participants
    }
    leader, *peers = participants
    event_symbol_actions = {}
    for peer in peers:
        arrive = canonical_plan_barrier_event_symbol(
            dag.id,
            PlanBarrierEventPhase.ARRIVE,
            peer,
            leader,
        )
        release = canonical_plan_barrier_event_symbol(
            dag.id,
            PlanBarrierEventPhase.RELEASE,
            leader,
            peer,
        )
        event_symbol_actions[arrive.id] = (peer, leader)
        event_symbol_actions[release.id] = (leader, peer)

    runtime_definitions = []
    for symbol in fragment.runtime_symbols:
        if symbol.kind is RuntimeSymbolKind.RUNTIME_CORE:
            action = core_symbol_actions[symbol.id]
            runtime_definitions.append(
                RuntimeSymbolDefinition(
                    symbol, (action.logical_core,), None, None
                )
            )
        else:
            source, destination = event_symbol_actions[symbol.id]
            runtime_definitions.append(
                RuntimeSymbolDefinition(
                    symbol,
                    tuple(
                        sorted(
                            (source.logical_core, destination.logical_core),
                            key=lambda core: (core.die_id, core.local_core_id),
                        )
                    ),
                    source.id,
                    destination.id,
                )
            )

    active_cores = tuple(
        sorted(
            (action.logical_core for action in participants),
            key=lambda core: (core.die_id, core.local_core_id),
        )
    )
    start_symbols = tuple(
        RuntimeSymbol(
            f"plan_barrier_start_{core.die_id}_{core.local_core_id}",
            RuntimeSymbolKind.START_TAG,
            next(
                action.id
                for action in participants
                if action.logical_core == core
            ),
        )
        for core in active_cores
    )
    runtime_definitions.extend(
        RuntimeSymbolDefinition(symbol, (core,), None, None)
        for symbol, core in zip(start_symbols, active_cores)
    )

    core_bindings = []
    core_streams = []
    for core in active_cores:
        die = next(item for item in ir1.fabric.dies if item.id == core.die_id)
        core_spec = next(
            item for item in die.cores if item.local_core_id == core.local_core_id
        )
        core_bindings.append(
            CoreRuntimeBinding(
                core,
                core_spec.id,
                core_spec.runtime_core_id,
                core_spec.sram_profile_ref,
            )
        )
        stream = next(
            item for item in fragment.core_streams if item.logical_core == core
        )
        core_streams.append(
            LinkedCoreStream(
                core,
                core_spec.runtime_core_id,
                tuple(
                    LinkedRecordRef(fragment.id, index, record.source_global_action_id)
                    for index, record in enumerate(stream.records)
                ),
            )
        )

    entry_symbols = []
    exit_symbols = []
    for stream in fragment.core_streams:
        for record in stream.records:
            tag = next(operand for operand in record.operands if operand.name == "tag")
            if record.opcode is RecordOpcode.EVENT_WAIT:
                entry_symbols.append(tag.symbol_ref)
            else:
                exit_symbols.append(tag.symbol_ref)
    interface = FragmentInterface(
        fragment.id,
        (),
        tuple(symbol.id for symbol in fragment.runtime_symbols),
        (),
        (),
        tuple(EventCredit(symbol_id, 1) for symbol_id in sorted(entry_symbols)),
        tuple(EventCredit(symbol_id, 1) for symbol_id in sorted(exit_symbols)),
    )
    envelope = ProgramControlEnvelope(
        active_cores,
        tuple(
            LogicalStartEvent(core, symbol.id, 1)
            for core, symbol in zip(active_cores, start_symbols)
        ),
        active_cores,
        active_cores,
        active_cores,
        EmptyCoreAckPolicy.INCLUDE_EMPTY,
        ProgramFailurePolicy.ABORT_ALL,
    )
    artifacts = (
        (ManifestInputKind.IR1, ir1),
        *((ManifestInputKind.FUSION_PLAN, plan) for plan in fusion_plans),
        *((ManifestInputKind.STANDALONE_PLAN, plan) for plan in standalone_plans),
        (ManifestInputKind.IR2_PROJECTION, projection),
        (ManifestInputKind.SCHEDULE_SET, schedule_set),
        (ManifestInputKind.GLOBAL_ACTION_DAG, dag),
        (ManifestInputKind.COMMAND_FRAGMENT, fragment),
    )
    manifest = LinkedProgramManifest.create(
        producer_pass="plan_barrier_linked_fixture",
        capabilities=0,
        source_ir1_id=ir1.id,
        source_projection_id=projection.id,
        source_schedule_set_id=schedule_set.id,
        source_global_dag_id=dag.id,
        input_digests=tuple(
            sorted(
                (
                    ManifestInputDigest(
                        kind,
                        artifact.id,
                        artifact.schema_version,
                        canonical_digest(artifact),
                    )
                    for kind, artifact in artifacts
                ),
                key=lambda item: (item.kind.value, item.artifact_id),
            )
        ),
        fragments=(fragment,),
        fragment_interfaces=(interface,),
        core_bindings=tuple(core_bindings),
        core_streams=tuple(core_streams),
        runtime_symbol_definitions=tuple(
            sorted(runtime_definitions, key=lambda item: item.symbol.id)
        ),
        program_symbol_definitions=(),
        address_operand_bindings=(),
        core_groups=(),
        envelope=envelope,
    )
    return (
        ir1,
        fusion_plans,
        standalone_plans,
        projection,
        schedule_set,
        dag,
        manifest,
    )


class LinkedProgramManifestSchemaTest(unittest.TestCase):
    def test_fused_recv_wait_token_definition_closes_owner_consumer_and_core(self) -> None:
        _ir1, _projection, _schedule_set, dag, manifest = valid_linked_manifest(
            with_recv_wait=True
        )
        manifest.validate()
        fragments = tuple(
            linked.fragment if hasattr(linked, "fragment") else linked
            for linked in manifest.fragments
        )
        actions = {action.id: action for action in dag.actions}
        definitions = {
            definition.symbol.id: definition
            for definition in manifest.runtime_symbol_definitions
        }
        _validate_fused_recv_wait_token_closure(
            actions, fragments, definitions, "runtime_definitions"
        )

        recv = next(
            action for action in dag.actions if action.task_kind.value == "recv"
        )
        wait = next(
            action for action in dag.actions if action.task_kind.value == "wait"
        )
        token = next(
            definition
            for definition in definitions.values()
            if definition.symbol.kind is RuntimeSymbolKind.DTE_TOKEN
        )
        self.assertEqual(token.logical_cores, (recv.logical_core,))
        self.assertEqual(token.source_action_id, recv.id)
        self.assertEqual(token.destination_action_id, wait.id)

        for changed in (
            replace(token, source_action_id=wait.id),
            replace(token, destination_action_id=None),
            replace(
                token,
                logical_cores=(
                    next(
                        core
                        for core in manifest.envelope.active_cores
                        if core != recv.logical_core
                    ),
                ),
            ),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                SchemaError, "owner, consumer and core"
            ):
                _validate_fused_recv_wait_token_closure(
                    actions,
                    fragments,
                    {**definitions, token.symbol.id: changed},
                    "runtime_definitions",
                )

        fragment = fragments[0]
        wait_stream_index = next(
            index
            for index, stream in enumerate(fragment.core_streams)
            if any(record.source_global_action_id == wait.id for record in stream.records)
        )
        wait_stream = fragment.core_streams[wait_stream_index]
        wait_record_index = next(
            index
            for index, record in enumerate(wait_stream.records)
            if record.source_global_action_id == wait.id
        )
        missing_wait_token = _recreate(
            fragment,
            core_streams=tuple(
                replace(
                    stream,
                    runtime_relocations=tuple(
                        relocation
                        for relocation in stream.runtime_relocations
                        if not (
                            relocation.record_index == wait_record_index
                            and relocation.field is RuntimeOperandField.DTE_TOKEN
                        )
                    ),
                )
                if index == wait_stream_index
                else stream
                for index, stream in enumerate(fragment.core_streams)
            ),
        )
        with self.assertRaisesRegex(SchemaError, "share one exact"):
            _validate_fused_recv_wait_token_closure(
                actions,
                (missing_wait_token,),
                definitions,
                "runtime_definitions",
            )

    def test_sram_bind_labels_close_activation_output_and_are_nonphysical(self) -> None:
        context, manifest = valid_exact_linked_manifest()
        for fragment in manifest.fragments:
            stream = fragment.core_streams[0]
            self.assertEqual(
                tuple(record.opcode for record in stream.records),
                (RecordOpcode.SRAM_BIND, RecordOpcode.MATMUL),
            )
            self.assertEqual(
                sum(
                    relocation.symbol_kind is ProgramSymbolKind.SRAM_LABEL
                    for relocation in stream.address_relocations
                ),
                2,
            )
            self.assertEqual(
                sum(
                    relocation.symbol_kind
                    is ProgramSymbolKind.ABSOLUTE_ADDRESS
                    for relocation in stream.address_relocations
                ),
                3,
            )
        labels = tuple(
            definition
            for definition in manifest.program_symbol_definitions
            if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
        )
        self.assertTrue(labels)
        self.assertTrue(
            all(
                (definition.value, definition.size_bytes) == (0, 0)
                for definition in labels
            )
        )

        activation = next(
            binding
            for binding in manifest.address_operand_bindings
            if binding.operand_id is SemanticOperandId.SRAM_BIND_INPUT_0
        )
        weight = next(
            binding
            for binding in manifest.address_operand_bindings
            if binding.fragment_id == activation.fragment_id
            and binding.logical_core == activation.logical_core
            and binding.operand_id is SemanticOperandId.COMPUTE_DATA_ADDRESS
        )
        bindings = tuple(
            replace(item, buffer_abi_ids=weight.buffer_abi_ids)
            if item == activation
            else item
            for item in manifest.address_operand_bindings
        )
        wrong_abi = _recreate_manifest(
            manifest, address_operand_bindings=bindings
        )
        with self.assertRaisesRegex(SchemaError, "operand roles"):
            wrong_abi.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                wrong_abi.fragments,
            )

        other_core = next(
            binding.logical_core
            for binding in manifest.core_bindings
            if binding.logical_core != activation.logical_core
        )
        wrong_core_bindings = tuple(
            sorted(
                (
                    replace(activation, logical_core=other_core)
                    if item == activation
                    else item
                    for item in manifest.address_operand_bindings
                ),
                key=lambda item: (
                    item.logical_core.die_id,
                    item.logical_core.local_core_id,
                    item.fragment_id,
                    item.fragment_record_index,
                    int(item.operand_id),
                ),
            )
        )
        with self.assertRaisesRegex(SchemaError, "requires one BufferABI closure"):
            _recreate_manifest(
                manifest, address_operand_bindings=wrong_core_bindings
            ).validate()

    def test_structural_round_trip_and_stable_digest(self) -> None:
        _ir1, _projection, _schedule_set, _dag, manifest = valid_linked_manifest()
        manifest.validate()
        decoded = loads_dataclass(LinkedProgramManifest, canonical_json(manifest))
        self.assertEqual(decoded, manifest)
        self.assertEqual(canonical_digest(decoded), canonical_digest(manifest))

    def test_exact_ordinary_context_cross_validates_and_round_trips(self) -> None:
        context, manifest = valid_exact_linked_manifest()
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        decoded = loads_dataclass(LinkedProgramManifest, canonical_json(manifest))
        self.assertEqual(decoded, manifest)
        self.assertEqual(canonical_digest(decoded), canonical_digest(manifest))

    def test_exact_input_digest_tamper_is_rejected(self) -> None:
        context, manifest = valid_exact_linked_manifest()
        tampered = replace(manifest.input_digests[0], digest="0" * 64)
        broken = _recreate_manifest(
            manifest,
            input_digests=(tampered, *manifest.input_digests[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "input digests"):
            broken.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                broken.fragments,
            )

    def test_compute_data_operand_cannot_alias_activation_closure(self) -> None:
        context, manifest = valid_exact_linked_manifest()
        activation = next(
            binding
            for binding in manifest.address_operand_bindings
            if binding.operand_id is SemanticOperandId.COMPUTE_INPUT_ADDRESS
        )
        data_index = next(
            index
            for index, binding in enumerate(manifest.address_operand_bindings)
            if binding.logical_core == activation.logical_core
            and binding.operand_id is SemanticOperandId.COMPUTE_DATA_ADDRESS
        )
        bindings = list(manifest.address_operand_bindings)
        bindings[data_index] = replace(
            bindings[data_index], buffer_abi_ids=activation.buffer_abi_ids
        )
        broken = _recreate_manifest(
            manifest, address_operand_bindings=tuple(bindings)
        )
        with self.assertRaisesRegex(SchemaError, "operand roles"):
            broken.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                broken.fragments,
            )

    def test_absolute_program_symbol_must_cover_full_operand_closure(self) -> None:
        context, manifest = valid_exact_linked_manifest()
        definition = next(
            item
            for item in manifest.program_symbol_definitions
            if item.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
        )
        broken_definition = replace(definition, size_bytes=1)
        definitions = tuple(
            broken_definition if item == definition else item
            for item in manifest.program_symbol_definitions
        )
        broken = _recreate_manifest(
            manifest, program_symbol_definitions=definitions
        )
        with self.assertRaisesRegex(SchemaError, "does not resolve to BufferABI"):
            broken.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                broken.fragments,
            )

    def test_program_symbol_name_utf8_and_inclusive_uint64_span(self) -> None:
        core = LogicalCoreRef(0, 0)
        symbol = ProgramSymbol(
            "absolute_max", ProgramSymbolKind.ABSOLUTE_ADDRESS, "fixture"
        )
        valid = ProgramSymbolDefinition(
            symbol, "max_byte", (1 << 64) - 1, 1, (core,)
        )
        valid.validate("definition")
        with self.assertRaisesRegex(SchemaError, "overflows"):
            replace(valid, value=(1 << 64) - 2, size_bytes=3).validate(
                "definition"
            )
        for bad_name, message in (
            ("bad\x00name", "NUL"),
            ("\ud800", "surrogate"),
            ("é" * 128, "255-byte"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(SchemaError, message):
                    replace(valid, name=bad_name).validate("definition")

    def test_local_reduce_requires_exact_named_region_containment_witness(self) -> None:
        core = LogicalCoreRef(0, 0)
        symbol = ProgramSymbol(
            "reduce_region", ProgramSymbolKind.SRAM_REGION, "reduce"
        )
        definition = ProgramSymbolDefinition(
            symbol, "sram", 0, 1 << 20, (core,)
        )
        definitions = {symbol.id: definition}
        self.assertEqual(
            _validate_local_reduce_containment_witness(
                (symbol,),
                definitions,
                core,
                "sram",
                0,
                1 << 20,
                32,
                96,
                "definition",
            ),
            symbol.id,
        )
        with self.assertRaisesRegex(SchemaError, "containment witness"):
            _validate_local_reduce_containment_witness(
                (symbol,),
                definitions,
                core,
                "wrong_region",
                0,
                1 << 20,
                32,
                96,
                "definition",
            )

    def test_address_operand_rejects_duplicate_abi_but_preserves_order(self) -> None:
        ordered = AddressOperandBinding(
            "fragment",
            LogicalCoreRef(0, 0),
            0,
            SemanticOperandId.SOURCE_ADDRESS,
            ("z_rank0", "a_rank1"),
            (
                TensorSlice("value", (0,), (1,)),
                TensorSlice("value", (1,), (1,)),
            ),
        )
        ordered.validate("binding")
        self.assertEqual(ordered.buffer_abi_ids, ("z_rank0", "a_rank1"))
        with self.assertRaisesRegex(SchemaError, "cannot repeat"):
            replace(ordered, buffer_abi_ids=("z_rank0", "z_rank0")).validate(
                "binding"
            )
        with self.assertRaisesRegex(SchemaError, "bijectively match"):
            replace(ordered, tensor_slices=ordered.tensor_slices[:1]).validate(
                "binding"
            )

    def test_reduce_input_closure_uses_rank_order_not_lexical_id_order(self) -> None:
        action, record, closure, by_binding, by_id = _reduce_operand_case()
        ordered = _validate_address_operand_closure(
            closure,
            action,
            record,
            BufferUseRole.REDUCE_INPUT,
            -1,
            by_binding,
            by_id,
            "binding",
        )
        self.assertEqual(
            tuple(abi.id for abi in ordered[0]), ("z_rank0", "a_rank1")
        )
        with self.assertRaisesRegex(SchemaError, "tensor views"):
            _validate_address_operand_closure(
                replace(
                    closure,
                    tensor_slices=tuple(reversed(closure.tensor_slices)),
                ),
                action,
                record,
                BufferUseRole.REDUCE_INPUT,
                -1,
                by_binding,
                by_id,
                "binding",
            )
        with self.assertRaisesRegex(SchemaError, "operand roles"):
            _validate_address_operand_closure(
                replace(closure, buffer_abi_ids=tuple(reversed(closure.buffer_abi_ids))),
                action,
                record,
                BufferUseRole.REDUCE_INPUT,
                -1,
                by_binding,
                by_id,
                "binding",
            )

    def test_reduce_input_closure_rejects_missing_rank_and_noncontiguous_span(self) -> None:
        action, record, closure, by_binding, by_id = _reduce_operand_case()
        missing_rank = replace(
            action,
            buffer_uses=tuple(
                use
                for use in action.buffer_uses
                if use.contribution_rank != 1
            ),
        )
        with self.assertRaisesRegex(SchemaError, "bijectively cover"):
            _validate_address_operand_closure(
                closure,
                missing_rank,
                record,
                BufferUseRole.REDUCE_INPUT,
                -1,
                by_binding,
                by_id,
                "binding",
            )
        rank1_key = next(
            key for key, abi in by_binding.items() if abi.id == "a_rank1"
        )
        noncontiguous = dict(by_binding)
        noncontiguous[rank1_key] = replace(
            noncontiguous[rank1_key], region_offset_bytes=64
        )
        with self.assertRaisesRegex(SchemaError, "contiguous rank-major"):
            _validate_address_operand_closure(
                closure,
                action,
                record,
                BufferUseRole.REDUCE_INPUT,
                -1,
                noncontiguous,
                {**by_id, "a_rank1": noncontiguous[rank1_key]},
                "binding",
            )

    def test_local_reduce_source_and_destination_starts_require_2byte_alignment(self) -> None:
        _validate_local_reduce_absolute_alignment((0, 32), "source_closure")
        _validate_local_reduce_absolute_alignment((64,), "destination_closure")
        for starts, path in (
            ((1, 33), "source_closure"),
            ((65,), "destination_closure"),
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(SchemaError, "2-byte aligned"):
                    _validate_local_reduce_absolute_alignment(starts, path)

    def test_runtime_symbol_binding_ref_rejects_forged_channel_token_event(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        dag = _create_global(ir1, projection, schedule_set)
        send = next(
            action
            for action in dag.actions
            if action.task_kind.value == "send"
        )
        recv = next(
            action
            for action in dag.actions
            if action.task_kind.value == "recv"
        )
        assert send.runtime_binding is not None
        assert send.logical_core is not None and recv.logical_core is not None
        cores = tuple(
            sorted(
                (send.logical_core, recv.logical_core),
                key=lambda core: (core.die_id, core.local_core_id),
            )
        )
        cases = (
            (
                RuntimeOperandField.DTE_FSM,
                RuntimeSymbolDefinition(
                    RuntimeSymbol(
                        "fsm",
                        RuntimeSymbolKind.DTE_FSM,
                        send.runtime_binding.channel_symbol,
                    ),
                    cores,
                    send.id,
                    recv.id,
                ),
            ),
            (
                RuntimeOperandField.DTE_TOKEN,
                RuntimeSymbolDefinition(
                    RuntimeSymbol(
                        "token",
                        RuntimeSymbolKind.DTE_TOKEN,
                        send.runtime_binding.token_symbol,
                    ),
                    (send.logical_core,),
                    send.id,
                    None,
                ),
            ),
            (
                RuntimeOperandField.EVENT_TAG,
                RuntimeSymbolDefinition(
                    RuntimeSymbol(
                        "event",
                        RuntimeSymbolKind.EVENT_TAG,
                        send.runtime_binding.event_symbol,
                    ),
                    cores,
                    send.id,
                    recv.id,
                ),
            ),
        )
        for field, definition in cases:
            with self.subTest(field=field):
                _validate_runtime_binding_ref(
                    definition, send, field, "runtime_definition"
                )
                forged = replace(
                    definition,
                    symbol=replace(definition.symbol, source_ref="forged_binding"),
                )
                with self.assertRaisesRegex(SchemaError, "schedule logical binding"):
                    _validate_runtime_binding_ref(
                        forged, send, field, "runtime_definition"
                    )
        event_definition = cases[-1][1]
        aligned_destination = replace(
            recv,
            runtime_binding=replace(
                recv.runtime_binding,
                event_symbol=send.runtime_binding.event_symbol,
            ),
        )
        _validate_runtime_binding_ref(
            event_definition,
            aligned_destination,
            RuntimeOperandField.EVENT_TAG,
            "runtime_definition",
        )
        with self.assertRaisesRegex(SchemaError, "schedule logical binding"):
            _validate_runtime_binding_ref(
                event_definition,
                recv,
                RuntimeOperandField.EVENT_TAG,
                "runtime_definition",
            )

    def test_cross_fragment_event_import_export_and_credit_round_trip(self) -> None:
        manifest = valid_cross_fragment_event_manifest()
        manifest.validate()
        decoded = loads_dataclass(LinkedProgramManifest, canonical_json(manifest))
        self.assertEqual(decoded, manifest)
        wait_fragment = next(
            fragment
            for fragment in manifest.fragments
            if fragment.core_streams[0].records[0].opcode
            is RecordOpcode.EVENT_WAIT
        )
        wait_record = wait_fragment.core_streams[0].records[0]
        for bad_count in (0, 1 << 32):
            with self.subTest(bad_count=bad_count):
                with self.assertRaisesRegex(SchemaError, "non-zero uint32"):
                    replace(
                        wait_record,
                        operands=(
                            *wait_record.operands[:-1],
                            replace(
                                wait_record.operands[-1],
                                literal_value=bad_count,
                            ),
                        ),
                    ).validate("event_wait")

    def test_cross_fragment_event_conflicting_declaration_is_rejected(self) -> None:
        manifest = valid_cross_fragment_event_manifest()
        consumer_interface = next(
            interface
            for interface in manifest.fragment_interfaces
            if interface.runtime_imports
        )
        consumer = next(
            fragment
            for fragment in manifest.fragments
            if fragment.id == consumer_interface.fragment_id
        )
        symbols = tuple(
            replace(symbol, source_ref="forged_event_binding")
            if symbol.kind is RuntimeSymbolKind.EVENT_TAG
            else symbol
            for symbol in consumer.runtime_symbols
        )
        forged_consumer = _recreate(consumer, runtime_symbols=symbols)
        fragments = tuple(
            sorted(
                (
                    forged_consumer
                    if fragment.id == consumer.id
                    else fragment
                    for fragment in manifest.fragments
                ),
                key=lambda item: item.id,
            )
        )
        interfaces = tuple(
            sorted(
                (
                    replace(interface, fragment_id=forged_consumer.id)
                    if interface.fragment_id == consumer.id
                    else interface
                    for interface in manifest.fragment_interfaces
                ),
                key=lambda item: item.fragment_id,
            )
        )
        streams = tuple(
            replace(
                stream,
                records=tuple(
                    replace(record, fragment_id=forged_consumer.id)
                    if record.fragment_id == consumer.id
                    else record
                    for record in stream.records
                ),
            )
            for stream in manifest.core_streams
        )
        broken = _recreate_manifest(
            manifest,
            fragments=fragments,
            fragment_interfaces=interfaces,
            core_streams=streams,
        )
        with self.assertRaisesRegex(SchemaError, "identical global definition"):
            broken.validate()

    def test_cross_fragment_event_duplicate_export_and_credit_tamper_rejected(self) -> None:
        manifest = valid_cross_fragment_event_manifest()
        consumer_index = next(
            index
            for index, interface in enumerate(manifest.fragment_interfaces)
            if interface.runtime_imports
        )
        interfaces = list(manifest.fragment_interfaces)
        consumer = interfaces[consumer_index]
        interfaces[consumer_index] = replace(
            consumer,
            runtime_imports=(),
            runtime_exports=consumer.runtime_imports,
        )
        with self.assertRaisesRegex(SchemaError, "exactly one exporting"):
            _recreate_manifest(
                manifest, fragment_interfaces=tuple(interfaces)
            ).validate()
        interfaces[consumer_index] = replace(
            consumer,
            entry_events=(EventCredit("event_shared", 2),),
        )
        with self.assertRaisesRegex(SchemaError, "exactly derive"):
            _recreate_manifest(
                manifest, fragment_interfaces=tuple(interfaces)
            ).validate()

    def test_cross_fragment_event_rejects_duplicate_set_wait_pairs(self) -> None:
        manifest = valid_cross_fragment_event_manifest()
        replacements = {}
        for fragment in manifest.fragments:
            stream = fragment.core_streams[0]
            duplicated = _recreate(
                fragment,
                core_streams=(
                    replace(
                        stream,
                        records=(stream.records[0], stream.records[0]),
                        runtime_relocations=stream.runtime_relocations
                        + tuple(
                            replace(relocation, record_index=1)
                            for relocation in stream.runtime_relocations
                        ),
                    ),
                ),
            )
            replacements[fragment.id] = duplicated
        fragments = tuple(
            sorted(replacements.values(), key=lambda item: item.id)
        )
        interfaces = tuple(
            sorted(
                (
                    replace(
                        interface,
                        fragment_id=replacements[interface.fragment_id].id,
                        entry_events=tuple(
                            replace(credit, count=2)
                            for credit in interface.entry_events
                        ),
                        exit_events=tuple(
                            replace(credit, count=2)
                            for credit in interface.exit_events
                        ),
                    )
                    for interface in manifest.fragment_interfaces
                ),
                key=lambda item: item.fragment_id,
            )
        )
        streams = tuple(
            replace(
                stream,
                records=tuple(
                    LinkedRecordRef(
                        replacements[record.fragment_id].id,
                        index,
                        record.source_global_action_id,
                    )
                    for record in stream.records
                    for index in (0, 1)
                ),
            )
            for stream in manifest.core_streams
        )
        duplicated = _recreate_manifest(
            manifest,
            fragments=fragments,
            fragment_interfaces=interfaces,
            core_streams=streams,
        )
        with self.assertRaisesRegex(SchemaError, "exactly one EVENT_SET"):
            duplicated.validate()

    def test_plan_barrier_linked_closure_is_exact_for_tp2_and_tp4(self) -> None:
        for tp in (2, 4):
            with self.subTest(tp=tp):
                fixture = _plan_barrier_linked_fixture(tp)
                (
                    ir1,
                    fusion_plans,
                    standalone_plans,
                    projection,
                    schedule_set,
                    dag,
                    manifest,
                ) = fixture
                manifest.validate()
                with patch.object(
                    GlobalActionDAG, "validate_against", return_value=None
                ):
                    manifest.validate_against(
                        ir1,
                        fusion_plans,
                        standalone_plans,
                        projection,
                        schedule_set,
                        dag,
                        manifest.fragments,
                    )
                event_definitions = tuple(
                    definition
                    for definition in manifest.runtime_symbol_definitions
                    if definition.symbol.kind is RuntimeSymbolKind.EVENT_TAG
                )
                self.assertEqual(len(event_definitions), 2 * (tp - 1))
                self.assertTrue(
                    all(
                        definition.symbol.source_ref
                        == dag.actions[0].sync.barrier.id
                        for definition in event_definitions
                    )
                )

    def test_plan_barrier_linked_rejects_endpoint_and_core_tamper(self) -> None:
        (
            ir1,
            fusion_plans,
            standalone_plans,
            projection,
            schedule_set,
            dag,
            manifest,
        ) = _plan_barrier_linked_fixture(2)
        event = next(
            definition
            for definition in manifest.runtime_symbol_definitions
            if definition.symbol.kind is RuntimeSymbolKind.EVENT_TAG
        )
        reversed_event = replace(
            event,
            source_action_id=event.destination_action_id,
            destination_action_id=event.source_action_id,
        )
        core_definition = next(
            definition
            for definition in manifest.runtime_symbol_definitions
            if definition.symbol.kind is RuntimeSymbolKind.RUNTIME_CORE
        )
        wrong_core = next(
            core
            for core in manifest.envelope.active_cores
            if core not in core_definition.logical_cores
        )
        cases = (
            reversed_event,
            replace(core_definition, logical_cores=(wrong_core,)),
        )
        for changed in cases:
            with self.subTest(kind=changed.symbol.kind):
                broken = _recreate_manifest(
                    manifest,
                    runtime_symbol_definitions=tuple(
                        changed
                        if definition.symbol.id == changed.symbol.id
                        else definition
                        for definition in manifest.runtime_symbol_definitions
                    ),
                )
                broken.validate()
                with patch.object(
                    GlobalActionDAG, "validate_against", return_value=None
                ), self.assertRaisesRegex(
                    SchemaError, "coordinator|dependency direction"
                ):
                    broken.validate_against(
                        ir1,
                        fusion_plans,
                        standalone_plans,
                        projection,
                        schedule_set,
                        dag,
                        broken.fragments,
                    )


if __name__ == "__main__":
    unittest.main()
