"""Require real shared-backbone gradient bytes before MoE backward dispatch.

The builder admits only a physical rank0 backward producer with a true FP16
token×H gradient output and then emits a blocking bounded local SRAM copy
into the named MoE backward-gradient carrier.  Its DTE copy is not a Dense
backward replacement: every norm/attention/loss/parameter WGRAD producer must
exist and be audited separately before claiming a full training model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.moe_full_training_namespace import NamespacedMoeTrainingUnit
from ..schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, BufferABI, CommandFragment,
    CoreFragmentStream, FragmentKind, LinkedProgramManifest,
    ProgramSymbol, ProgramSymbolDefinition, ProgramSymbolKind, RecordOpcode,
    RecordOperand, RelocatableRecord, RuntimeOperandField,
    RuntimeRelocation, RuntimeSymbol, RuntimeSymbolDefinition,
    RuntimeSymbolKind, SemanticOperandId,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.ir2 import dense_row_major_view_byte_addend
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileUnit


_SCHEMA = "wafer_frontend.moe_training_shared_gradient_bridge/v1alpha1"
_CORE = LogicalCoreRef(0, 0)


@dataclass(frozen=True, slots=True)
class SharedBackwardGradientProducer:
    source_action_id: str
    buffer: BufferABI
    address_definition: ProgramSymbolDefinition
    valid_bytes: int


@dataclass(frozen=True, slots=True)
class MoeTrainingSharedGradientBridge:
    fragment: CommandFragment
    runtime_definition: RuntimeSymbolDefinition
    program_definitions: tuple[ProgramSymbolDefinition, ProgramSymbolDefinition]
    address_bindings: tuple[AddressOperandBinding, AddressOperandBinding]
    source_action_id: str
    moe_gradient_send_action_id: str
    valid_bytes: int


def locate_shared_backward_gradient_producer(
    manifest: LinkedProgramManifest,
    *,
    source_action_id: str,
    expected_opcode: RecordOpcode,
    tokens: int,
    hidden_size: int,
) -> SharedBackwardGradientProducer:
    """Refuse optimizer motif output or an unbound/oversized CE gradient.

    Producer must bind one actual COMPUTE_OUTPUT_ADDRESS view of exactly
    token_count×hidden_size FP16; later full-model graph coverage independently
    proves this action is the named reverse residual/backbone operation.
    """
    manifest.validate("moe_shared_backward_producer_source")
    if (not source_action_id or type(expected_opcode) is not RecordOpcode
            or tokens <= 0 or hidden_size <= 0):
        raise SchemaError("shared gradient needs typed native producer and dimensions",
                          path="source_action_id")
    physical_bytes = tokens * hidden_size * 2
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    buffers = {abi.id: abi for fragment in manifest.fragments
               for abi in fragment.buffer_abi}
    defs = {definition.symbol.id: definition for definition
            in manifest.program_symbol_definitions}
    closure = {(binding.fragment_id, binding.logical_core,
                binding.fragment_record_index, binding.operand_id): binding
               for binding in manifest.address_operand_bindings}
    found = []
    for stream in manifest.core_streams:
        if stream.logical_core != _CORE:
            continue
        for ref in stream.records:
            if ref.source_global_action_id != source_action_id:
                continue
            leaf = fragments[ref.fragment_id]
            local = next(item for item in leaf.core_streams
                         if item.logical_core == _CORE)
            record = local.records[ref.fragment_record_index]
            if record.opcode is not expected_opcode:
                continue
            binding = closure.get((ref.fragment_id, _CORE,
                                   ref.fragment_record_index,
                                   SemanticOperandId.COMPUTE_OUTPUT_ADDRESS))
            if binding is None or len(binding.buffer_abi_ids) != 1:
                raise SchemaError("shared backward output lacks one exact BufferABI",
                                  path=source_action_id)
            abi = buffers[binding.buffer_abi_ids[0]]
            if (abi.logical_core != _CORE or abi.dtype is not DType.FP16
                    or abi.size_bytes < physical_bytes
                    or abi.alias_of is not None):
                raise SchemaError("source shared gradient is not owned FP16 m×H SRAM",
                                  path=source_action_id)
            view = binding.tensor_slices[0]
            if (view.value_id != abi.value_id or
                    view.shape != (tokens, hidden_size) or
                    dense_row_major_view_byte_addend(
                        abi.tensor_slice, view, abi.dtype,
                        path="shared_backward_gradient_view") != 0):
                raise SchemaError("shared backward output does not cover exact token×H view",
                                  path=source_action_id)
            operands = [item for item in record.operands if
                        item.operand_id is SemanticOperandId.COMPUTE_OUTPUT_ADDRESS]
            if len(operands) != 1 or operands[0].symbol_ref not in defs:
                raise SchemaError("shared backward producer has no physical output symbol",
                                  path=source_action_id)
            definition = defs[operands[0].symbol_ref]
            if (definition.symbol.kind is not ProgramSymbolKind.ABSOLUTE_ADDRESS
                    or definition.symbol.source_ref != abi.binding_id
                    or definition.size_bytes != abi.size_bytes):
                raise SchemaError("producer output symbol differs from its actual SRAM home",
                                  path=source_action_id)
            found.append(SharedBackwardGradientProducer(
                source_action_id, abi, definition, physical_bytes))
    if len(found) != 1:
        raise SchemaError("one named real shared backbone backward producer is required",
                          path=source_action_id)
    return found[0]


def build_moe_training_shared_gradient_bridge(
    unit: MoeCompileUnit,
    named: NamespacedMoeTrainingUnit,
    producer: SharedBackwardGradientProducer,
    *,
    source_global_dag_id: str,
) -> MoeTrainingSharedGradientBridge:
    """Copy the exact FP16 model gradient before source-rank MoE dispatch.

    DTE_ISSUE is an actual local SRAM→SRAM transfer followed by DTE_WAIT;
    source/target ABSOLUTE_ADDRESS symbols and buffer closures are both rooted
    in real allocated regions.  The composed timeline must place this action
    after producer action and before rank0 BACKWARD_GRADIENT DTE_SEND.
    """
    if (unit.id != named.source_unit_id or unit.step != named.step
            or unit.layer != named.layer or named.source_manifest_id !=
            unit.linked_manifest.id or named.fragments[0].source_global_dag_id !=
            source_global_dag_id):
        raise SchemaError("shared/MoE bridge needs identical real step/layer Global DAG",
                          path="source_global_dag_id")
    if producer.buffer.logical_core != _CORE or producer.buffer.dtype is not DType.FP16:
        raise SchemaError("only physical rank0 FP16 shared reverse output can dispatch",
                          path="producer.buffer")
    bytes_required = unit.spec.trace.token_count * unit.spec.hidden_size * 2
    if producer.valid_bytes != bytes_required:
        raise SchemaError("producer width must match the declared model token×H",
                          path="producer.valid_bytes")
    dest = [abi for fragment in named.fragments for abi in fragment.buffer_abi
            if abi.logical_core == _CORE and
            abi.value_id.endswith(".rank0.backward_gradient")]
    target = {abi.id: abi for abi in dest}
    if len(target) != 1:
        raise SchemaError("one real rank0 backward-gradient SRAM carrier is required",
                          path="named.fragments.buffer_abi")
    dest_abi = next(iter(target.values()))
    if dest_abi.dtype is not DType.FP16 or dest_abi.size_bytes < bytes_required:
        raise SchemaError("MoE source gradient carrier is narrower than declared model",
                          path="dest_abi")
    backward = [flow for flow in unit.plan.flows
                if flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT
                and flow.source_rank == 0 and flow.destination_rank == 1]
    if len(backward) != 1:
        raise SchemaError("source MoE backward gradient must dispatch rank0→rank1",
                          path="unit.plan.flows")
    send = [action for action in unit.plan.actions
            if action.kind is MoeRectActionKind.SEND and
            action.flow_ref == backward[0].id and action.rank == 0]
    if len(send) != 1 or send[0].logical_bytes > bytes_required:
        raise SchemaError("rank0 physical backward SEND lacks its P2 bytes",
                          path="unit.plan.actions")
    mapped_send = dict(named.action_ids)[send[0].id]
    if not any(record.source_global_action_id == mapped_send and
               record.opcode is RecordOpcode.DTE_SEND
               for fragment in named.fragments for local in fragment.core_streams
               for record in local.records):
        raise SchemaError("gradient bridge would feed no actual MoE DTE SEND",
                          path="mapped_send")
    dest_definition = [definition for definition in named.program_definitions
                       if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                       and definition.symbol.source_ref == dest_abi.binding_id]
    if len(dest_definition) != 1 or dest_definition[0].size_bytes != dest_abi.size_bytes:
        raise SchemaError("MoE gradient destination lacks one exact absolute home",
                          path=dest_abi.id)
    source_home = producer.address_definition.value
    dest_home = dest_definition[0].value
    if (source_home < 0 or dest_home < 0 or
            source_home + bytes_required > producer.address_definition.value
                                         + producer.buffer.size_bytes or
            dest_home + bytes_required > dest_definition[0].value
                                       + dest_abi.size_bytes):
        raise SchemaError("gradient bridge crosses physical producer/target backing",
                          path="gradient_bridge_home")
    action = stable_artifact_id(
        "moe_training_shared_gradient_bridge_action",
        {"dag": source_global_dag_id, "step": unit.step, "layer": unit.layer,
         "producer": producer.source_action_id, "send": mapped_send,
         "bytes": bytes_required}, schema_version=_SCHEMA,
    )
    token = RuntimeSymbol(stable_artifact_id(
        "moe_training_shared_gradient_copy_token", {"action": action},
        schema_version=_SCHEMA), RuntimeSymbolKind.DTE_TOKEN, action)
    defs, symbols = [], []
    for role, abi, value in (("source", producer.buffer, source_home),
                             ("destination", dest_abi, dest_home)):
        symbol = ProgramSymbol(stable_artifact_id(
            "moe_training_shared_gradient_address",
            {"action": action, "role": role, "binding": abi.binding_id},
            schema_version=_SCHEMA), ProgramSymbolKind.ABSOLUTE_ADDRESS,
            abi.binding_id)
        defs.append(ProgramSymbolDefinition(symbol, f"moe.train.step{unit.step}.layer{unit.layer}.{role}.gradient", value, abi.size_bytes, (_CORE,)))
        symbols.append(symbol)
    records = (
        RelocatableRecord(action, RecordOpcode.DTE_ISSUE, (
            RecordOperand.literal("direction", 0),
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
            RecordOperand.literal("payload_bits", bytes_required * 8),
            RecordOperand.literal("size_bytes", bytes_required),
            RecordOperand.literal("hbm_address", 0),
            RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS,
                                  symbols[0].id),
            RecordOperand.address("destination_address",
                                  SemanticOperandId.DESTINATION_ADDRESS,
                                  symbols[1].id),
        )),
        RelocatableRecord(action, RecordOpcode.DTE_WAIT, (
            RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
        )),
    )
    fragment = CommandFragment.create(
        producer_pass="moe_training_shared_gradient_bridge",
        source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(action,),
        core_streams=(CoreFragmentStream(
            _CORE, records,
            (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
             RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id)),
            (AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS,
                               ProgramSymbolKind.ABSOLUTE_ADDRESS, symbols[0].id, 0),
             AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS,
                               ProgramSymbolKind.ABSOLUTE_ADDRESS, symbols[1].id, 0)),
        ),), runtime_symbols=(token,), program_symbols=tuple(symbols),
        buffer_abi=(producer.buffer, dest_abi), state_abi=(),
    )
    fragment.validate("moe_training_gradient_bridge_fragment")
    bindings = (
        AddressOperandBinding(fragment.id, _CORE, 0,
                              SemanticOperandId.SOURCE_ADDRESS,
                              (producer.buffer.id,),
                              (producer.buffer.tensor_slice,)),
        AddressOperandBinding(fragment.id, _CORE, 0,
                              SemanticOperandId.DESTINATION_ADDRESS,
                              (dest_abi.id,), (dest_abi.tensor_slice,)),
    )
    return MoeTrainingSharedGradientBridge(
        fragment,
        RuntimeSymbolDefinition(token, (_CORE,), action, None),
        tuple(defs), bindings, producer.source_action_id, mapped_send,
        bytes_required,
    )


__all__ = ["SharedBackwardGradientProducer", "MoeTrainingSharedGradientBridge",
           "locate_shared_backward_gradient_producer",
           "build_moe_training_shared_gradient_bridge"]
