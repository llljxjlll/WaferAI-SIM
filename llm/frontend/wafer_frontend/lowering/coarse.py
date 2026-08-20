"""Deterministic coarse lowering for the currently executable Dense subset."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.action import ComputeContract
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
    SemanticOperandId,
    _ComputeRecordABI,
    _compute_record_abi,
    _FIXED_COMPUTE_OPCODES,
    _fixed_compute_literals,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import GlobalAction, LogicalCoreRef
from ..schema.ir0 import GemmWorkload
from ..schema.ir2 import (
    BufferBinding,
    BufferOwnership,
    BufferUseRole,
    RegionLowering,
    SemanticTaskKind,
    dense_row_major_view_byte_addend,
)
from .context import LoweringContext


_PRODUCER_PASS = "coarse_lowering"


def _buffer_abi(
    schedule_id: str,
    binding: BufferBinding,
    logical_core: LogicalCoreRef,
) -> BufferABI:
    semantic = {
        "schedule_id": schedule_id,
        "binding_id": binding.id,
        "value_id": binding.value_id,
        "logical_core": logical_core,
        "tensor_slice": binding.tensor_slice,
        "region_ref": binding.region_ref,
        "region_offset_bytes": binding.region_offset_bytes,
        "size_bytes": binding.size_bytes,
        "alignment_bytes": binding.alignment_bytes,
        "banks": binding.banks,
        "storage_id": binding.storage_id,
        "alias_of": binding.alias_of,
        "lifetime_start": binding.lifetime_start,
        "lifetime_end_exclusive": binding.lifetime_end_exclusive,
        "dtype": binding.dtype,
        "layout": binding.layout,
        "ownership": binding.ownership,
    }
    return BufferABI(
        id=stable_artifact_id(
            "buffer_abi",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        **semantic,
    )


def _program_symbol(
    *,
    schedule_id: str,
    binding: BufferBinding,
    kind: ProgramSymbolKind,
) -> ProgramSymbol:
    if kind is ProgramSymbolKind.SRAM_LABEL:
        semantic = {
            "schedule_id": schedule_id,
            "runtime_core_id": binding.core_id,
            "storage_id": binding.storage_id,
            "kind": int(kind),
        }
        source_ref = binding.storage_id
    else:
        semantic = {
            "schedule_id": schedule_id,
            "binding_id": binding.id,
            "kind": int(kind),
        }
        source_ref = binding.id
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        kind,
        source_ref,
    )


def _binding_for_use(
    action: GlobalAction,
    bindings: dict[str, BufferBinding],
    role: BufferUseRole,
    operand_index: int,
    *,
    path: str,
) -> BufferBinding:
    matches = tuple(
        use
        for use in action.buffer_uses
        if use.role is role and use.operand_index == operand_index
    )
    if len(matches) != 1 or matches[0].binding_id not in bindings:
        raise SchemaError(
            "requires exactly one scheduled buffer binding for the operand",
            path=path,
        )
    return bindings[matches[0].binding_id]


def _view_addend_for_use(
    action: GlobalAction,
    binding: BufferBinding,
    role: BufferUseRole,
    operand_index: int,
    *,
    path: str,
) -> int:
    matches = tuple(
        use
        for use in action.buffer_uses
        if use.role is role and use.operand_index == operand_index
    )
    if len(matches) != 1 or matches[0].binding_id != binding.id:
        raise SchemaError(
            "requires exactly one matching action buffer view",
            path=path,
        )
    return dense_row_major_view_byte_addend(
        binding.tensor_slice,
        matches[0].tensor_slice,
        binding.dtype,
        path=path,
    )


def _require_compute(
    action: GlobalAction,
    path: str,
    *,
    lowering: RegionLowering = RegionLowering.JSON_COARSE,
) -> tuple[ComputeContract, _ComputeRecordABI]:
    compute = action.compute
    if (
        action.task_kind is not SemanticTaskKind.COMP
        or action.lowering is not lowering
        or action.logical_core is None
        or compute is None
        or compute.tile is not None
    ):
        raise SchemaError(
            "coarse lowering requires one ordinary scheduled Dense COMP",
            path=path,
        )
    return compute, _compute_record_abi(compute, path=f"{path}.compute")


def _fixed_compute_operands(
    compute: ComputeContract,
    abi: _ComputeRecordABI,
    input_address: ProgramSymbol,
    data_address: ProgramSymbol | None,
    aux_address: ProgramSymbol | None,
    output_address: ProgramSymbol,
) -> tuple[RecordOperand, ...]:
    values = _fixed_compute_literals(compute, abi.opcode, path="action.compute")
    if abi.opcode is RecordOpcode.ROPE_QK_EXACT:
        return (
            RecordOperand.literal("datatype", values["datatype"]),
            RecordOperand.literal("packed_layout", values["packed_layout"]),
            RecordOperand.address(
                "input_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "output_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logical_tokens",
                    "tp_degree",
                    "num_heads",
                    "num_kv_heads",
                    "rank_num_heads",
                    "rank_num_kv_heads",
                    "head_dim",
                    "rotary_dim",
                    "max_position_embeddings",
                    "context_max",
                    "rope_theta_f64_bits",
                )
            ),
        )
    if abi.opcode is RecordOpcode.ATTENTION_EXACT:
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in ("datatype", "mode", "packed_layout", "causal")
            ),
            RecordOperand.address(
                "input_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "output_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "query_tokens",
                    "tp_degree",
                    "num_heads",
                    "num_kv_heads",
                    "rank_num_heads",
                    "rank_num_kv_heads",
                    "head_dim",
                    "context_sum",
                    "context_max",
                    "query_key_pairs",
                    "rank_kv_read_bytes",
                    "rank_kv_write_bytes",
                )
            ),
        )
    if abi.opcode is RecordOpcode.EMBEDDING_LOOKUP:
        if data_address is None:
            raise SchemaError(
                "EMBEDDING_LOOKUP requires a table DATA address",
                path="action.compute",
            )
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "index_datatype",
                    "table_datatype",
                    "output_datatype",
                    "placement",
                )
            ),
            RecordOperand.address(
                "indices_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "table_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                data_address.id,
            ),
            RecordOperand.address(
                "output_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logical_rows",
                    "rank_rows",
                    "tp_degree",
                    "vocab_size",
                    "hidden_size",
                )
            ),
        )
    if abi.opcode is RecordOpcode.GREEDY_SAMPLE:
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logits_datatype",
                    "output_datatype",
                    "mode",
                    "row_selection",
                )
            ),
            RecordOperand.address(
                "logits_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "output_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "tp_degree",
                    "token_rows",
                    "vocab_size",
                    "sample_count",
                    "comparisons",
                )
            ),
        )
    if abi.opcode is RecordOpcode.CROSS_ENTROPY_FORWARD:
        if data_address is None:
            raise SchemaError(
                "CROSS_ENTROPY_FORWARD requires a labels DATA address",
                path="action.compute",
            )
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logits_datatype",
                    "label_datatype",
                    "loss_datatype",
                    "reduction",
                )
            ),
            RecordOperand.address(
                "logits_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "labels_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                data_address.id,
            ),
            RecordOperand.address(
                "loss_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logical_rows",
                    "rank_rows",
                    "tp_degree",
                    "vocab_size",
                )
            ),
        )
    if abi.opcode is RecordOpcode.CROSS_ENTROPY_BACKWARD:
        if data_address is None or aux_address is None:
            raise SchemaError(
                "CROSS_ENTROPY_BACKWARD requires labels DATA and upstream AUX addresses",
                path="action.compute",
            )
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logits_datatype",
                    "label_datatype",
                    "upstream_datatype",
                    "output_datatype",
                    "reduction",
                    "upstream_mode",
                )
            ),
            RecordOperand.address(
                "logits_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "labels_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                data_address.id,
            ),
            RecordOperand.address(
                "upstream_address",
                SemanticOperandId.COMPUTE_AUX_ADDRESS,
                aux_address.id,
            ),
            RecordOperand.address(
                "logits_grad_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "logical_rows",
                    "rank_rows",
                    "tp_degree",
                    "vocab_size",
                    "upstream_elements",
                )
            ),
        )
    if abi.opcode is RecordOpcode.SGD_UPDATE:
        if data_address is None:
            raise SchemaError(
                "SGD_UPDATE requires a gradient DATA address",
                path="action.compute",
            )
        return (
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "weight_datatype",
                    "gradient_datatype",
                    "output_datatype",
                    "rounding",
                )
            ),
            RecordOperand.address(
                "weight_address",
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                input_address.id,
            ),
            RecordOperand.address(
                "gradient_address",
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                data_address.id,
            ),
            RecordOperand.address(
                "updated_weight_address",
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                output_address.id,
            ),
            *(
                RecordOperand.literal(name, values[name])
                for name in (
                    "element_count",
                    "learning_rate_f64_bits",
                    "momentum_f64_bits",
                )
            ),
        )
    raise SchemaError("opcode is not a fixed compute record", path="action.compute")


def _require_matmul(
    action: GlobalAction,
    path: str,
    *,
    lowering: RegionLowering = RegionLowering.JSON_COARSE,
) -> tuple[ComputeContract, GemmWorkload]:
    """Compatibility contract used by the fused ISA-region lowerer."""

    compute = action.compute
    if (
        action.task_kind is not SemanticTaskKind.COMP
        or action.lowering is not lowering
        or action.logical_core is None
        or compute is None
    ):
        raise SchemaError(
            "lowering requires scheduled FP16 matmul_forward GEMM",
            path=path,
        )
    abi = _compute_record_abi(compute, path=f"{path}.compute")
    if (
        abi.opcode is not RecordOpcode.MATMUL
        or type(compute.workload) is not GemmWorkload
    ):
        raise SchemaError(
            "lowering requires scheduled FP16 matmul_forward GEMM",
            path=path,
        )
    return compute, compute.workload


class NaiveCoarseLowering:
    """Lower one ordinary Dense compute without changing schedule decisions."""

    def __init__(self, *, validate_output: bool = True) -> None:
        self._validate_output = validate_output
        self._validated_contexts: list[LoweringContext] = []

    def _validate_context_once(self, context: LoweringContext) -> None:
        if not any(previous is context for previous in self._validated_contexts):
            context.validate()
            self._validated_contexts.append(context)

    def lower(
        self,
        action: GlobalAction,
        context: LoweringContext,
    ) -> CommandFragment:
        if type(context) is not LoweringContext:
            raise SchemaError("must be a LoweringContext", path="context")
        if type(action) is not GlobalAction:
            raise SchemaError("must be a GlobalAction", path="action")
        self._validate_context_once(context)
        source = next(
            (candidate for candidate in context.global_dag.actions if candidate.id == action.id),
            None,
        )
        if source is None or source != action:
            raise SchemaError(
                "action must exactly equal one action in the lowering context",
                path="action",
            )
        compute, compute_abi = _require_compute(action, "action")
        assert action.logical_core is not None

        schedule = next(
            (
                candidate
                for candidate in context.schedule_set.schedules
                if candidate.id == action.source.schedule_id
            ),
            None,
        )
        if schedule is None:
            raise SchemaError(
                "action references an unknown schedule",
                path="action.source.schedule_id",
            )
        bindings = {binding.id: binding for binding in schedule.buffer_bindings}
        inputs = tuple(
            _binding_for_use(
                action,
                bindings,
                BufferUseRole.COMP_INPUT,
                operand_index,
                path="action.buffer_uses",
            )
            for operand_index in range(len(compute.inputs))
        )
        output = _binding_for_use(
            action,
            bindings,
            BufferUseRole.COMP_OUTPUT,
            0,
            path="action.buffer_uses",
        )
        input_addends = tuple(
            _view_addend_for_use(
                action,
                binding,
                BufferUseRole.COMP_INPUT,
                operand_index,
                path="action.buffer_uses",
            )
            for operand_index, binding in enumerate(inputs)
        )
        output_addend = _view_addend_for_use(
            action,
            output,
            BufferUseRole.COMP_OUTPUT,
            0,
            path="action.buffer_uses",
        )

        bound_inputs = inputs[: compute_abi.bind_input_count]
        input_labels = tuple(
            _program_symbol(
                schedule_id=schedule.id,
                binding=binding,
                kind=ProgramSymbolKind.SRAM_LABEL,
            )
            for binding in bound_inputs
        )
        output_label = _program_symbol(
            schedule_id=schedule.id,
            binding=output,
            kind=ProgramSymbolKind.SRAM_LABEL,
        )
        input_address = _program_symbol(
            schedule_id=schedule.id,
            binding=inputs[0],
            kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
        )
        data_address = (
            _program_symbol(
                schedule_id=schedule.id,
                binding=inputs[compute_abi.data_input_index],
                kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
            )
            if compute_abi.data_input_index is not None
            else None
        )
        aux_address = (
            _program_symbol(
                schedule_id=schedule.id,
                binding=inputs[compute_abi.aux_input_index],
                kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
            )
            if compute_abi.aux_input_index is not None
            else None
        )
        output_address = _program_symbol(
            schedule_id=schedule.id,
            binding=output,
            kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
        )
        if compute_abi.opcode is RecordOpcode.SGD_UPDATE:
            weight = inputs[0]
            same_binding = output.id == weight.id
            exact_alias = (
                output.ownership is BufferOwnership.ALIASED
                and output.alias_of == weight.id
                and weight.ownership is not BufferOwnership.ALIASED
                and output.core_id == weight.core_id
                and output.region_ref == weight.region_ref
                and output.region_offset_bytes == weight.region_offset_bytes
                and output.size_bytes == weight.size_bytes
                and output.alignment_bytes == weight.alignment_bytes
                and output.banks == weight.banks
                and output.storage_id == weight.storage_id
                and output.tensor_slice.offset == weight.tensor_slice.offset
                and output.tensor_slice.shape == weight.tensor_slice.shape
                and output.dtype is weight.dtype
                and output.layout == weight.layout
            )
            if (
                (not same_binding and not exact_alias)
                or output_addend != input_addends[0]
            ):
                raise SchemaError(
                    "SGD_UPDATE output must be the exact weight binding or its exact derived alias/view",
                    path="action.buffer_uses",
                )
            output_address = input_address

        bind = RelocatableRecord(
            action.id,
            RecordOpcode.SRAM_BIND,
            (
                RecordOperand.literal(
                    "input_count", compute_abi.bind_input_count
                ),
                *(
                    RecordOperand.address(
                        f"input_label_{index}",
                        SemanticOperandId(
                            int(SemanticOperandId.SRAM_BIND_INPUT_0) + index
                        ),
                        input_labels[index].id,
                    )
                    if index < compute_abi.bind_input_count
                    else RecordOperand.literal(f"input_label_{index}", 0)
                    for index in range(16)
                ),
                RecordOperand.address(
                    "output_label",
                    SemanticOperandId.SRAM_BIND_OUTPUT,
                    output_label.id,
                ),
            ),
        )
        compute_record = RelocatableRecord(
            action.id,
            compute_abi.opcode,
            (
                _fixed_compute_operands(
                    compute,
                    compute_abi,
                    input_address,
                    data_address,
                    aux_address,
                    output_address,
                )
                if compute_abi.opcode in _FIXED_COMPUTE_OPCODES
                else (
                    RecordOperand.literal("datatype", 1),
                    RecordOperand.address(
                        "input_address",
                        SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                        input_address.id,
                    ),
                    (
                        RecordOperand.address(
                            "data_address",
                            SemanticOperandId.COMPUTE_DATA_ADDRESS,
                            data_address.id,
                        )
                        if data_address is not None
                        else RecordOperand.literal("data_address", 0)
                    ),
                    RecordOperand.address(
                        "output_address",
                        SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                        output_address.id,
                    ),
                    RecordOperand.literal("parameters", compute_abi.parameters),
                )
            ),
        )
        bind_relocations = tuple(
            AddressRelocation(
                0,
                SemanticOperandId(
                    int(SemanticOperandId.SRAM_BIND_INPUT_0) + index
                ),
                ProgramSymbolKind.SRAM_LABEL,
                label.id,
                0,
            )
            for index, label in enumerate(input_labels)
        ) + (
            AddressRelocation(
                0,
                SemanticOperandId.SRAM_BIND_OUTPUT,
                ProgramSymbolKind.SRAM_LABEL,
                output_label.id,
                0,
            ),
        )
        compute_relocations = (
            AddressRelocation(
                1,
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                input_address.id,
                input_addends[0],
            ),
            *(
                (
                    AddressRelocation(
                        1,
                        SemanticOperandId.COMPUTE_DATA_ADDRESS,
                        ProgramSymbolKind.ABSOLUTE_ADDRESS,
                        data_address.id,
                        input_addends[compute_abi.data_input_index],
                    ),
                )
                if data_address is not None
                else ()
            ),
            *(
                (
                    AddressRelocation(
                        1,
                        SemanticOperandId.COMPUTE_AUX_ADDRESS,
                        ProgramSymbolKind.ABSOLUTE_ADDRESS,
                        aux_address.id,
                        input_addends[compute_abi.aux_input_index],
                    ),
                )
                if aux_address is not None
                and compute_abi.aux_input_index is not None
                else ()
            ),
            AddressRelocation(
                1,
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                ProgramSymbolKind.ABSOLUTE_ADDRESS,
                output_address.id,
                output_addend,
            ),
        )
        stream = CoreFragmentStream(
            action.logical_core,
            (bind, compute_record),
            (),
            bind_relocations + compute_relocations,
        )
        used_bindings = (*inputs, output)
        program_symbols = (
            *input_labels,
            output_label,
            input_address,
            *((data_address,) if data_address is not None else ()),
            *((aux_address,) if aux_address is not None else ()),
            output_address,
        )
        fragment = CommandFragment.create(
            producer_pass=_PRODUCER_PASS,
            source_global_dag_id=context.global_dag.id,
            kind=FragmentKind.COARSE,
            claimed_action_ids=(action.id,),
            core_streams=(stream,),
            runtime_symbols=(),
            program_symbols=tuple(
                sorted(
                    {symbol.id: symbol for symbol in program_symbols}.values(),
                    key=lambda symbol: symbol.id,
                )
            ),
            buffer_abi=tuple(
                sorted(
                    {
                        binding.id: _buffer_abi(
                            schedule.id, binding, action.logical_core
                        )
                        for binding in used_bindings
                    }.values(),
                    key=lambda abi: abi.id,
                )
            ),
        )
        if self._validate_output:
            fragment.validate_against(context.global_dag)
        return fragment
