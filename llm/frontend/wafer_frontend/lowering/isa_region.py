"""Deterministic ISA lowering for one NAIVE fused GEMM->ReduceScatter plan."""

from __future__ import annotations

from collections.abc import Iterable

from ..errors import SchemaError
from ..schema.action import CollectiveAlgorithm, FusionPlan
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    CommandFragment,
    CoreFragmentStream,
    FragmentKind,
    ProgramSymbol,
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
    _fused_recv_wait_pairs,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import GlobalAction, LogicalCoreRef
from ..schema.ir0 import FusionImpl
from ..schema.ir2 import (
    BufferBinding,
    BufferUseRole,
    FusedNodeOrigin,
    RegionLowering,
    SemanticTaskKind,
    dense_row_major_view_byte_addend,
)
from .coarse import (
    _binding_for_use,
    _buffer_abi,
    _program_symbol,
    _require_matmul,
    _view_addend_for_use,
)
from .context import LoweringContext


_PRODUCER_PASS = "isa_region_lowering"
_RUNTIME_FIELD_ORDER = {
    field: index for index, field in enumerate(RuntimeOperandField)
}


def _runtime_symbol(
    kind: RuntimeSymbolKind, source_ref: str, identity: object
) -> RuntimeSymbol:
    semantic = {
        "kind": kind.value,
        "source_ref": source_ref,
        "identity": identity,
    }
    return RuntimeSymbol(
        stable_artifact_id(
            "runtime_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        kind,
        source_ref,
    )


def _multi_binding_program_symbol(
    *,
    schedule_id: str,
    bindings: tuple[BufferBinding, ...],
    operand: SemanticOperandId,
) -> ProgramSymbol:
    if not bindings:
        raise SchemaError("requires non-empty buffer bindings", path="bindings")
    semantic = {
        "schedule_id": schedule_id,
        "binding_ids": tuple(binding.id for binding in bindings),
        "kind": int(ProgramSymbolKind.ABSOLUTE_ADDRESS),
        "operand": int(operand),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.ABSOLUTE_ADDRESS,
        bindings[0].id,
    )


def _transport_symbols(action: GlobalAction) -> tuple[RuntimeSymbol, RuntimeSymbol]:
    binding = action.runtime_binding
    if (
        binding is None
        or binding.channel_symbol is None
        or action.flow_id is None
    ):
        raise SchemaError(
            "transport action requires its exact N5 channel runtime binding",
            path="action.runtime_binding",
        )
    fsm = _runtime_symbol(
        RuntimeSymbolKind.DTE_FSM,
        binding.channel_symbol,
        ("channel", binding.channel_symbol),
    )
    peer = _runtime_symbol(
        RuntimeSymbolKind.RUNTIME_CORE,
        binding.channel_symbol,
        ("peer", action.id),
    )
    return fsm, peer


def _token_symbol(recv: GlobalAction) -> RuntimeSymbol:
    binding = recv.runtime_binding
    if binding is None or binding.token_symbol is None:
        raise SchemaError(
            "waited RECV requires its exact N5 token runtime binding",
            path="action.runtime_binding.token_symbol",
        )
    return _runtime_symbol(
        RuntimeSymbolKind.DTE_TOKEN,
        binding.token_symbol,
        ("recv", recv.id),
    )


def _matmul_records(
    action: GlobalAction,
    schedule_id: str,
    bindings: dict[str, BufferBinding],
) -> tuple[
    tuple[RelocatableRecord, ...],
    tuple[ProgramSymbol, ...],
    tuple[AddressRelocation, ...],
    tuple[BufferBinding, ...],
]:
    _compute, workload = _require_matmul(
        action, "action", lowering=RegionLowering.ISA_REGION
    )
    activation = _binding_for_use(
        action, bindings, BufferUseRole.COMP_INPUT, 0, path="action.buffer_uses"
    )
    weight = _binding_for_use(
        action, bindings, BufferUseRole.COMP_INPUT, 1, path="action.buffer_uses"
    )
    output = _binding_for_use(
        action, bindings, BufferUseRole.COMP_OUTPUT, 0, path="action.buffer_uses"
    )
    activation_addend = _view_addend_for_use(
        action, activation, BufferUseRole.COMP_INPUT, 0,
        path="action.buffer_uses",
    )
    weight_addend = _view_addend_for_use(
        action, weight, BufferUseRole.COMP_INPUT, 1,
        path="action.buffer_uses",
    )
    output_addend = _view_addend_for_use(
        action, output, BufferUseRole.COMP_OUTPUT, 0,
        path="action.buffer_uses",
    )
    activation_label = _program_symbol(
        schedule_id=schedule_id,
        binding=activation,
        kind=ProgramSymbolKind.SRAM_LABEL,
    )
    output_label = _program_symbol(
        schedule_id=schedule_id,
        binding=output,
        kind=ProgramSymbolKind.SRAM_LABEL,
    )
    activation_address = _program_symbol(
        schedule_id=schedule_id,
        binding=activation,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    weight_address = _program_symbol(
        schedule_id=schedule_id,
        binding=weight,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    output_address = _program_symbol(
        schedule_id=schedule_id,
        binding=output,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    bind = RelocatableRecord(
        action.id,
        RecordOpcode.SRAM_BIND,
        (
            RecordOperand.literal("input_count", 1),
            RecordOperand.address(
                "input_label_0",
                SemanticOperandId.SRAM_BIND_INPUT_0,
                activation_label.id,
            ),
            *(
                RecordOperand.literal(f"input_label_{index}", 0)
                for index in range(1, 16)
            ),
            RecordOperand.address(
                "output_label",
                SemanticOperandId.SRAM_BIND_OUTPUT,
                output_label.id,
            ),
        ),
    )
    rank_m, rank_n, rank_k = workload.rank_shape
    matmul = RelocatableRecord(
        action.id,
        RecordOpcode.MATMUL,
        (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address(
                "input_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                activation_address.id,
            ),
            RecordOperand.address(
                "data_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                weight_address.id,
            ),
            RecordOperand.address(
                "output_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            RecordOperand.literal("parameters", (1, rank_m, rank_k, rank_n)),
        ),
    )
    relocations = (
        AddressRelocation(
            0,
            SemanticOperandId.SRAM_BIND_INPUT_0,
            ProgramSymbolKind.SRAM_LABEL,
            activation_label.id,
            0,
        ),
        AddressRelocation(
            0,
            SemanticOperandId.SRAM_BIND_OUTPUT,
            ProgramSymbolKind.SRAM_LABEL,
            output_label.id,
            0,
        ),
        AddressRelocation(
            1,
            SemanticOperandId.COMPUTE_INPUT_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            activation_address.id,
            activation_addend,
        ),
        AddressRelocation(
            1,
            SemanticOperandId.COMPUTE_DATA_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            weight_address.id,
            weight_addend,
        ),
        AddressRelocation(
            1,
            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            output_address.id,
            output_addend,
        ),
    )
    return (
        (bind, matmul),
        (
            activation_label,
            output_label,
            activation_address,
            weight_address,
            output_address,
        ),
        relocations,
        (activation, weight, output),
    )


def _transport_record(
    action: GlobalAction,
    schedule_id: str,
    bindings: dict[str, BufferBinding],
    *,
    token: RuntimeSymbol | None,
) -> tuple[
    RelocatableRecord,
    tuple[RuntimeSymbol, ...],
    ProgramSymbol,
    tuple[RuntimeRelocation, ...],
    AddressRelocation,
    BufferBinding,
]:
    fsm, peer = _transport_symbols(action)
    is_send = action.task_kind is SemanticTaskKind.SEND
    role = BufferUseRole.SEND_SOURCE if is_send else BufferUseRole.RECV_DESTINATION
    binding = _binding_for_use(
        action, bindings, role, 0, path="action.buffer_uses"
    )
    addend = _view_addend_for_use(
        action, binding, role, 0, path="action.buffer_uses"
    )
    address = _program_symbol(
        schedule_id=schedule_id,
        binding=binding,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    token_operand = (
        RecordOperand.literal("token", 0)
        if token is None
        else RecordOperand.runtime(
            "token", RuntimeOperandField.DTE_TOKEN, token.id
        )
    )
    if is_send:
        record = RelocatableRecord(
            action.id,
            RecordOpcode.DTE_SEND,
            (
                RecordOperand.literal("mode", 0),
                RecordOperand.literal("source_space", 0),
                RecordOperand.literal("completion", 1),
                RecordOperand.literal("datatype", 0),
                RecordOperand.literal("reduce_op", 0),
                RecordOperand.runtime(
                    "fsm_id", RuntimeOperandField.DTE_FSM, fsm.id
                ),
                token_operand,
                RecordOperand.literal("length_bytes", action.bytes),
                RecordOperand.address(
                    "source_address", SemanticOperandId.SOURCE_ADDRESS, address.id
                ),
                RecordOperand.runtime(
                    "peer_core", RuntimeOperandField.PEER_CORE, peer.id
                ),
                RecordOperand.literal("expected_sources", 0),
                RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0),
                RecordOperand.literal("collective_id", 0),
                RecordOperand.literal("epoch", 0),
            ),
        )
        operand_id = SemanticOperandId.SOURCE_ADDRESS
    else:
        if token is None:
            raise SchemaError(
                "fused RECV requires an async token",
                path="action.runtime_binding.token_symbol",
            )
        record = RelocatableRecord(
            action.id,
            RecordOpcode.DTE_RECV,
            (
                RecordOperand.literal("mode", 0),
                RecordOperand.literal("completion", 0),
                RecordOperand.literal("datatype", 0),
                RecordOperand.literal("reduce_op", 0),
                RecordOperand.runtime(
                    "fsm_id", RuntimeOperandField.DTE_FSM, fsm.id
                ),
                token_operand,
                RecordOperand.literal("length_bytes", action.bytes),
                RecordOperand.address(
                    "destination_address",
                    SemanticOperandId.DESTINATION_ADDRESS,
                    address.id,
                ),
                RecordOperand.runtime(
                    "peer_core", RuntimeOperandField.PEER_CORE, peer.id
                ),
                RecordOperand.literal("expected_sources", 0),
                RecordOperand.literal("tree_id", 0),
                RecordOperand.literal("group_id", 0),
                RecordOperand.literal("collective_id", 0),
                RecordOperand.literal("epoch", 0),
            ),
        )
        operand_id = SemanticOperandId.DESTINATION_ADDRESS
    runtime_relocations = [
        RuntimeRelocation(0, RuntimeOperandField.DTE_FSM, fsm.id),
        RuntimeRelocation(0, RuntimeOperandField.PEER_CORE, peer.id),
    ]
    runtime_symbols = [fsm, peer]
    if token is not None:
        runtime_relocations.append(
            RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id)
        )
        runtime_symbols.append(token)
    return (
        record,
        tuple(runtime_symbols),
        address,
        tuple(
            sorted(
                runtime_relocations,
                key=lambda item: _RUNTIME_FIELD_ORDER[item.field],
            )
        ),
        AddressRelocation(
            0,
            operand_id,
            ProgramSymbolKind.ABSOLUTE_ADDRESS,
            address.id,
            addend,
        ),
        binding,
    )


def _reduce_record(
    action: GlobalAction,
    schedule_id: str,
    bindings: dict[str, BufferBinding],
) -> tuple[
    RelocatableRecord,
    tuple[ProgramSymbol, ...],
    tuple[AddressRelocation, ...],
    tuple[BufferBinding, ...],
]:
    reduction = action.reduction
    legacy = (
        reduction is not None
        and action.dtype is DType.FP16
        and action.bytes > 0
        and action.bytes % 2 == 0
        and reduction.input_dtype is DType.FP16
        and reduction.accumulation_dtype is DType.FP32
        and reduction.output_dtype is DType.FP16
    )
    dp2_fp32 = (
        reduction is not None
        and action.dtype is DType.FP32
        and action.bytes == 2048
        and reduction.input_dtype is DType.FP32
        and reduction.accumulation_dtype is DType.FP32
        and reduction.output_dtype is DType.FP32
        and reduction.input_ranks == (0, 1)
    )
    if not (legacy or dp2_fp32):
        raise SchemaError(
            "LOCAL_REDUCE requires its exact FP16 contract or DP2 2048-byte FP32 contract",
            path="action.reduction",
        )
    dtype_literal = 1 if dp2_fp32 else 0
    element_bytes = 4 if dp2_fp32 else 2
    input_uses = tuple(
        sorted(
            (
                use
                for use in action.buffer_uses
                if use.role is BufferUseRole.REDUCE_INPUT
            ),
            key=lambda use: use.operand_index,
        )
    )
    if (
        tuple(use.operand_index for use in input_uses)
        != tuple(range(len(reduction.input_ranks)))
        or tuple(use.contribution_rank for use in input_uses)
        != reduction.input_ranks
        or any(use.binding_id not in bindings for use in input_uses)
    ):
        raise SchemaError(
            "LOCAL_REDUCE inputs must exactly follow reduction rank order",
            path="action.buffer_uses",
        )
    input_bindings = tuple(bindings[use.binding_id] for use in input_uses)
    output = _binding_for_use(
        action,
        bindings,
        BufferUseRole.REDUCE_OUTPUT,
        0,
        path="action.buffer_uses",
    )
    input_addends = tuple(
        dense_row_major_view_byte_addend(
            binding.tensor_slice,
            use.tensor_slice,
            binding.dtype,
            path="action.buffer_uses",
        )
        for use, binding in zip(input_uses, input_bindings)
    )
    output_addend = _view_addend_for_use(
        action, output, BufferUseRole.REDUCE_OUTPUT, 0,
        path="action.buffer_uses",
    )
    source = _multi_binding_program_symbol(
        schedule_id=schedule_id,
        bindings=input_bindings,
        operand=SemanticOperandId.SOURCE_ADDRESS,
    )
    destination = _program_symbol(
        schedule_id=schedule_id,
        binding=output,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    )
    record = RelocatableRecord(
        action.id,
        RecordOpcode.LOCAL_REDUCE,
        (
            RecordOperand.literal("input_dtype", dtype_literal),
            RecordOperand.literal("accumulator_dtype", 1),
            RecordOperand.literal("output_dtype", dtype_literal),
            RecordOperand.literal("reduce_op", 1),
            RecordOperand.literal("rounding", 0),
            RecordOperand.literal("order", 0),
            RecordOperand.literal("input_count", len(input_bindings)),
            RecordOperand.literal("element_count", action.bytes // element_bytes),
            RecordOperand.literal("input_stride_bytes", action.bytes),
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
    return (
        record,
        (source, destination),
        (
            AddressRelocation(
                0,
                SemanticOperandId.SOURCE_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                source.id,
                input_addends[0],
            ),
            AddressRelocation(
                0,
                SemanticOperandId.DESTINATION_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                destination.id,
                output_addend,
            ),
        ),
        (*input_bindings, output),
    )


def _exact_plan_actions(
    plan: FusionPlan, context: LoweringContext
) -> tuple[GlobalAction, ...]:
    return tuple(
        action
        for action in context.global_dag.actions
        if isinstance(action.origin_ref, FusedNodeOrigin)
        and action.origin_ref.plan_id == plan.id
        and action.task_kind is not SemanticTaskKind.TRANSIT
    )


def _unique_by_id(values: Iterable[object]) -> tuple[object, ...]:
    result: dict[str, object] = {}
    for value in values:
        value_id = getattr(value, "id")
        previous = result.setdefault(value_id, value)
        if previous != value:
            raise SchemaError("content-addressed id collision", path="lowering")
    return tuple(result[key] for key in sorted(result))


class NaiveIsaRegionLowering:
    """Lower one N4 NAIVE plan into canonical per-die ISA region manifests."""

    def __init__(self, *, validate_output: bool = True) -> None:
        self._validate_output = validate_output
        self._validated_contexts: list[LoweringContext] = []

    def _validate_context_once(self, context: LoweringContext) -> None:
        if not any(previous is context for previous in self._validated_contexts):
            context.validate()
            self._validated_contexts.append(context)

    def lower(
        self,
        plan: FusionPlan,
        actions: tuple[GlobalAction, ...],
        context: LoweringContext,
    ) -> tuple[RegionManifest, ...]:
        if type(context) is not LoweringContext:
            raise SchemaError("must be a LoweringContext", path="context")
        if type(plan) is not FusionPlan:
            raise SchemaError("must be a FusionPlan", path="plan")
        if type(actions) is not tuple or any(
            type(action) is not GlobalAction for action in actions
        ):
            raise SchemaError(
                "must be a tuple of GlobalAction", path="actions"
            )
        self._validate_context_once(context)
        source_plan = next(
            (candidate for candidate in context.fusion_plans if candidate.id == plan.id),
            None,
        )
        if source_plan is None or source_plan != plan:
            raise SchemaError(
                "plan must exactly equal one plan in the lowering context",
                path="plan",
            )
        plan.validate_against(context.ir1, "plan")
        if (
            plan.impl is not FusionImpl.NAIVE
            or plan.collective_algorithm is not CollectiveAlgorithm.DIRECT
        ):
            raise SchemaError(
                "ISA region v1 supports only NAIVE DIRECT fused GEMM->ReduceScatter",
                path="plan",
            )
        expected_actions = _exact_plan_actions(plan, context)
        if actions != expected_actions:
            raise SchemaError(
                "actions must exactly preserve the context plan action tuple",
                path="actions",
            )
        allowed = {
            SemanticTaskKind.COMP,
            SemanticTaskKind.SEND,
            SemanticTaskKind.RECV,
            SemanticTaskKind.WAIT,
            SemanticTaskKind.REDUCE,
        }
        if not actions or any(
            action.task_kind not in allowed
            or action.lowering is not RegionLowering.ISA_REGION
            or action.logical_core is None
            for action in actions
        ):
            raise SchemaError(
                "plan actions must be executable ISA GEMM/transport/wait/reduce actions",
                path="actions",
            )

        all_actions = {action.id: action for action in context.global_dag.actions}
        recv_by_wait, wait_by_recv = _fused_recv_wait_pairs(
            all_actions, "actions"
        )
        schedules = {
            schedule.id: schedule for schedule in context.schedule_set.schedules
        }
        by_die: dict[int, list[GlobalAction]] = {}
        for action in actions:
            assert action.logical_core is not None
            by_die.setdefault(action.logical_core.die_id, []).append(action)

        manifests: list[RegionManifest] = []
        for die_id in sorted(by_die):
            local_actions = tuple(by_die[die_id])
            region_ids = {action.region_id for action in local_actions}
            if len(region_ids) != 1:
                raise SchemaError(
                    "one die/plan must map to exactly one local ISA region",
                    path="actions.region_id",
                )
            region_id = next(iter(region_ids))
            schedule_ids = {action.source.schedule_id for action in local_actions}
            if len(schedule_ids) != 1:
                raise SchemaError(
                    "one local ISA region must use exactly one schedule",
                    path="actions.source.schedule_id",
                )
            schedule_id = next(iter(schedule_ids))
            schedule = schedules.get(schedule_id)
            if schedule is None or schedule.die_id != die_id:
                raise SchemaError(
                    "local ISA region references an unknown die schedule",
                    path="actions.source.schedule_id",
                )
            bindings = {
                binding.id: binding for binding in schedule.buffer_bindings
            }

            actions_by_core: dict[LogicalCoreRef, list[GlobalAction]] = {}
            for action in local_actions:
                assert action.logical_core is not None
                actions_by_core.setdefault(action.logical_core, []).append(action)
            streams: list[CoreFragmentStream] = []
            fragment_runtime_symbols: list[RuntimeSymbol] = []
            fragment_program_symbols: list[ProgramSymbol] = []
            fragment_bindings: list[BufferBinding] = []
            for core in sorted(
                actions_by_core,
                key=lambda item: (item.die_id, item.local_core_id),
            ):
                core_actions = tuple(
                    sorted(
                        actions_by_core[core],
                        key=lambda action: action.core_order_index,
                    )
                )
                records: list[RelocatableRecord] = []
                runtime_relocations: list[RuntimeRelocation] = []
                address_relocations: list[AddressRelocation] = []
                for action in core_actions:
                    record_base = len(records)
                    if action.task_kind is SemanticTaskKind.COMP:
                        (
                            emitted,
                            program_symbols,
                            emitted_addresses,
                            used_bindings,
                        ) = _matmul_records(action, schedule_id, bindings)
                        records.extend(emitted)
                        fragment_program_symbols.extend(program_symbols)
                        address_relocations.extend(
                            AddressRelocation(
                                relocation.record_index + record_base,
                                relocation.operand_id,
                                relocation.symbol_kind,
                                relocation.symbol_ref,
                                relocation.addend,
                            )
                            for relocation in emitted_addresses
                        )
                        fragment_bindings.extend(used_bindings)
                    elif action.task_kind in (
                        SemanticTaskKind.SEND,
                        SemanticTaskKind.RECV,
                    ):
                        token = (
                            _token_symbol(action)
                            if action.task_kind is SemanticTaskKind.RECV
                            and action.id in wait_by_recv
                            else None
                        )
                        (
                            record,
                            runtime_symbols,
                            program_symbol,
                            emitted_runtime,
                            emitted_address,
                            used_binding,
                        ) = _transport_record(
                            action,
                            schedule_id,
                            bindings,
                            token=token,
                        )
                        records.append(record)
                        fragment_runtime_symbols.extend(runtime_symbols)
                        fragment_program_symbols.append(program_symbol)
                        runtime_relocations.extend(
                            RuntimeRelocation(
                                record_base,
                                relocation.field,
                                relocation.symbol_ref,
                            )
                            for relocation in emitted_runtime
                        )
                        address_relocations.append(
                            AddressRelocation(
                                record_base,
                                emitted_address.operand_id,
                                emitted_address.symbol_kind,
                                emitted_address.symbol_ref,
                                emitted_address.addend,
                            )
                        )
                        fragment_bindings.append(used_binding)
                    elif action.task_kind is SemanticTaskKind.WAIT:
                        recv = recv_by_wait[action.id]
                        token = _token_symbol(recv)
                        records.append(
                            RelocatableRecord(
                                action.id,
                                RecordOpcode.DTE_WAIT,
                                (
                                    RecordOperand.runtime(
                                        "token",
                                        RuntimeOperandField.DTE_TOKEN,
                                        token.id,
                                    ),
                                ),
                            )
                        )
                        fragment_runtime_symbols.append(token)
                        runtime_relocations.append(
                            RuntimeRelocation(
                                record_base,
                                RuntimeOperandField.DTE_TOKEN,
                                token.id,
                            )
                        )
                    else:
                        (
                            record,
                            program_symbols,
                            emitted_addresses,
                            used_bindings,
                        ) = _reduce_record(action, schedule_id, bindings)
                        records.append(record)
                        fragment_program_symbols.extend(program_symbols)
                        address_relocations.extend(
                            AddressRelocation(
                                record_base,
                                relocation.operand_id,
                                relocation.symbol_kind,
                                relocation.symbol_ref,
                                relocation.addend,
                            )
                            for relocation in emitted_addresses
                        )
                        fragment_bindings.extend(used_bindings)
                streams.append(
                    CoreFragmentStream(
                        core,
                        tuple(records),
                        tuple(
                            sorted(
                                runtime_relocations,
                                key=lambda item: (
                                    item.record_index,
                                    _RUNTIME_FIELD_ORDER[item.field],
                                ),
                            )
                        ),
                        tuple(
                            sorted(
                                address_relocations,
                                key=lambda item: (
                                    item.record_index, int(item.operand_id)
                                ),
                            )
                        ),
                    )
                )

            runtime_symbols = _unique_by_id(fragment_runtime_symbols)
            program_symbols = _unique_by_id(fragment_program_symbols)
            used_bindings = _unique_by_id(fragment_bindings)
            fragment = CommandFragment.create(
                producer_pass=_PRODUCER_PASS,
                source_global_dag_id=context.global_dag.id,
                kind=FragmentKind.ISA_REGION,
                claimed_action_ids=tuple(
                    sorted(action.id for action in local_actions)
                ),
                core_streams=tuple(streams),
                runtime_symbols=runtime_symbols,
                program_symbols=program_symbols,
                buffer_abi=tuple(
                    sorted(
                        (
                            _buffer_abi(schedule_id, binding, core)
                            for binding in used_bindings
                            for core in (
                                next(
                                    action.logical_core
                                    for action in local_actions
                                    if any(
                                        use.binding_id == binding.id
                                        for use in action.buffer_uses
                                    )
                                ),
                            )
                        ),
                        key=lambda abi: abi.id,
                    )
                ),
            )
            if self._validate_output:
                fragment.validate_against(context.global_dag)
            manifest = RegionManifest.create(
                producer_pass=_PRODUCER_PASS,
                region_id=region_id,
                fusion_plan_id=plan.id,
                target_dies=(die_id,),
                fragment=fragment,
            )
            if self._validate_output:
                manifest.validate_against(context.global_dag)
            manifests.append(manifest)
        return tuple(manifests)


__all__ = ["NaiveIsaRegionLowering"]
