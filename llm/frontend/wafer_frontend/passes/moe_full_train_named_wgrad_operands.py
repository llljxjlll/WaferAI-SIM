"""Independent production-leaf oracle for three expert 0x25 physical operands.

Derive operand spans from linked forward/DGRAD/SwiGLU-backward records and
BufferABI definitions.  Reject MATMUL/cast substitution and misbound gate/up
upstream even when the record/action names remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, OperandKind, ProgramSymbolKind, RecordOpcode,
    SemanticOperandId,
)
from ..schema.common import DType
from ..schema.flexible_moe import MoeRectActionKind
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..lowering.flexible_moe_multi_production import (
    expert_dgrad_action_ids, expert_projection_action_ids,
    expert_wgrad_action_ids,
)
from .moe_full_train_gradient_source_bridge import (
    MoeFullTrainGradientSourceBridge,
)


@dataclass(frozen=True, slots=True)
class MoeWgradPhysicalSlice:
    buffer_abi_ref: str
    absolute_symbol_ref: str
    value_ref: str
    dtype: DType
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class MoeExpertPhysicalWgradOperandSource:
    layer: int
    expert: int
    projection: str
    source_forward_action_ref: str
    source_dgrad_action_ref: str
    source_backward_swiglu_action_ref: str
    source_wgrad_action_ref: str
    native_record_action_ref: str
    forward_activation: MoeWgradPhysicalSlice
    derivative_upstream: MoeWgradPhysicalSlice
    fp32_gradient_output: MoeWgradPhysicalSlice
    m: int
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class MoeFullTrainPhysicalWgradOperandSources:
    source_moe_sequence_ref: str
    source_gradient_bridge_ref: str
    entries: tuple[MoeExpertPhysicalWgradOperandSource, ...]

    def validate_against(self, bridge: MoeFullTrainGradientSourceBridge,
                         sequence: MoeCompileSequence) -> None:
        if (self.source_moe_sequence_ref != sequence.id
                or self.source_gradient_bridge_ref != bridge.source_ir0_ref
                or self.entries != _derive(bridge, sequence)):
            raise SchemaError("forward, backward SwiGLU and all named FP32 gradient operand slices must exactly match production BufferABI",
                              path="moe_full_train_physical_wgrad_operands")

    def require_gate_up_derivative_consumption(
        self, sequence: MoeCompileSequence,
    ) -> None:
        """Refuse actual old `(k,H)` backward gradient as `(k,I)` dGate/dUp."""
        self._require_same_source(sequence)
        for entry in self.entries:
            if entry.projection not in ("gate", "up"):
                continue
            manifest = _unit(sequence, entry.layer).linked_manifest
            record, fragment, stream, index = _one_record(
                manifest, entry.expert, entry.native_record_action_ref,
                (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, RecordOpcode.MATMUL),
            )
            role = ("upstream_address" if record.opcode is
                    RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING else "data_address")
            actual = _operand_slice(manifest, fragment, stream, index, role,
                                    entry.derivative_upstream.size_bytes)
            if actual != entry.derivative_upstream:
                raise SchemaError(
                    "gate/up WGRAD must consume two SwiGLU-backward `(k,I)` slices, not old H-shaped backward_by_rank",
                    path=f"moe_wgrad.layer{entry.layer}.expert{entry.expert}.{entry.projection}",
                )

    def require_native_fp32_producers(self,
                                      sequence: MoeCompileSequence) -> None:
        """Old FP16 MATMUL→LOCAL_REDUCE cannot pass as real 0x25 dW output."""
        self._require_same_source(sequence)
        for entry in self.entries:
            manifest = _unit(sequence, entry.layer).linked_manifest
            record, fragment, stream, index = _one_record(
                manifest, entry.expert, entry.native_record_action_ref,
                (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, RecordOpcode.MATMUL),
            )
            if record.opcode is not RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING:
                raise SchemaError(
                    "FP16 MATMUL then FP32 LOCAL_REDUCE cast cannot substitute native 0x25 physical gradient producer",
                    path=f"moe_wgrad.layer{entry.layer}.expert{entry.expert}.{entry.projection}",
                )
            literals = {item.name: item.literal_value for item in record.operands
                        if item.kind is OperandKind.LITERAL}
            if (tuple(literals.get(key) for key in ("m", "n", "k"))
                    != (entry.m, entry.n, entry.k)
                    or tuple(literals.get(key) for key in
                             ("activation_datatype", "upstream_datatype",
                              "gradient_datatype")) != (1, 1, 3)):
                raise SchemaError("native 0x25 source tile geometry/dtype differs from original E2E/P2",
                                  path=f"moe_wgrad.layer{entry.layer}.{entry.projection}")
            expected = (("activation_address", entry.forward_activation),
                        ("upstream_address", entry.derivative_upstream),
                        ("gradient_address", entry.fp32_gradient_output))
            if any(_operand_slice(manifest, fragment, stream, index, name,
                                  source.size_bytes, require_exact_view=True)
                   != source
                   for name, source in expected):
                raise SchemaError("native 0x25 operand symbol/relocation/BufferABI slice differs from source forward/backward",
                                  path=f"moe_wgrad.layer{entry.layer}.{entry.projection}")

    def _require_same_source(self, sequence: MoeCompileSequence) -> None:
        if (self.source_moe_sequence_ref != sequence.id
                or len(self.entries) != 12):
            raise SchemaError("missing source-bound expert operand entry",
                              path="moe_full_train_physical_wgrad_operands")


def _unit(sequence, layer):
    unit = next((unit for unit in sequence.units
                 if (unit.step, unit.layer) == (0, layer)), None)
    if unit is None:
        raise SchemaError("missing true first-step physical expert leaf",
                          path=f"moe_wgrad.layer{layer}")
    return unit


def _one_record(manifest, rank, action_ref, opcodes):
    found = [(record, fragment, stream, index)
             for fragment in manifest.fragments for stream in fragment.core_streams
             if stream.logical_core.die_id == rank
             for index, record in enumerate(stream.records)
             if record.source_global_action_id == action_ref
             and record.opcode in opcodes]
    if len(found) != 1:
        raise SchemaError("one exact source-backed physical record/action required",
                          path=f"moe_wgrad.action[{action_ref}]")
    return found[0]


def _buffer_slice(manifest, rank, suffix, dtype, offset, size):
    roots = {abi.id: abi for fragment in manifest.fragments
             for abi in fragment.buffer_abi
             if abi.logical_core.die_id == rank
             and abi.value_id.endswith(suffix)}
    if len(roots) != 1:
        raise SchemaError("actual physical BufferABI with one exact rank/source is required",
                          path=f"moe_wgrad.rank{rank}.{suffix}")
    abi = next(iter(roots.values()))
    if (abi.dtype is not dtype or offset < 0 or size <= 0
            or offset + size > abi.size_bytes):
        raise SchemaError("physical FP16/FP32 operand slice exceeds its real BufferABI",
                          path=f"moe_wgrad.rank{rank}.{suffix}")
    definitions = [definition for definition in manifest.program_symbol_definitions
                   if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                   and definition.symbol.source_ref == abi.binding_id
                   and any(core.die_id == rank
                           for core in definition.logical_cores)]
    if len(definitions) != 1 or definitions[0].size_bytes != abi.size_bytes:
        raise SchemaError("BufferABI has no unique production absolute SRAM symbol",
                          path=f"moe_wgrad.rank{rank}.{suffix}")
    return MoeWgradPhysicalSlice(abi.id, definitions[0].symbol.id,
                                 abi.value_id, abi.dtype, offset, size)


def _operand_slice(manifest, fragment, stream, index, name, size,
                   *, require_exact_view=False):
    operand = next((item for item in stream.records[index].operands
                    if item.name == name), None)
    if operand is None or operand.kind is not OperandKind.ADDRESS_SYMBOL:
        raise SchemaError("one named physical address operand required",
                          path=f"moe_wgrad.record[{index}].{name}")
    reloc = next((item for item in stream.address_relocations
                  if item.record_index == index
                  and item.operand_id is operand.operand_id), None)
    closure = next((item for item in manifest.address_operand_bindings
                    if item.fragment_id == fragment.id
                    and item.logical_core == stream.logical_core
                    and item.fragment_record_index == index
                    and item.operand_id is operand.operand_id), None)
    if (reloc is None or closure is None
            or reloc.symbol_ref != operand.symbol_ref
            or len(closure.buffer_abi_ids) != 1):
        raise SchemaError("operand relocation and one BufferABI closure must agree",
                          path=f"moe_wgrad.record[{index}].{name}")
    abi = next((abi for abi in fragment.buffer_abi
                if abi.id == closure.buffer_abi_ids[0]), None)
    if abi is None or abi.dtype not in (DType.FP16, DType.FP32):
        raise SchemaError("operand has no typed physical SRAM ABI",
                          path=f"moe_wgrad.record[{index}].{name}")
    element_bytes = 4 if abi.dtype is DType.FP32 else 2
    expected_shape = (size // element_bytes,)
    view = closure.tensor_slices[0]
    lower = view.offset[0] if len(view.offset) == 1 else -1
    extent = view.shape[0] if len(view.shape) == 1 else -1
    actual_lower = reloc.addend // element_bytes
    if (size % element_bytes or reloc.addend < 0
            or reloc.addend % element_bytes
            or view.value_id != abi.value_id
            or lower < 0 or extent <= 0
            or lower > actual_lower
            or lower + extent < actual_lower + expected_shape[0]
            or (require_exact_view and
                (lower != actual_lower or extent != expected_shape[0]))):
        raise SchemaError("operand tensor slice does not cover exact WGRAD span",
                          path=f"moe_wgrad.record[{index}].{name}")
    return MoeWgradPhysicalSlice(
        abi.id, operand.symbol_ref, abi.value_id, abi.dtype,
        reloc.addend, size,
    )


def _producer_matches(manifest, rank, action_ref, opcode, operand_name,
                      expected, size):
    record, fragment, stream, index = _one_record(
        manifest, rank, action_ref, (opcode,))
    if _operand_slice(manifest, fragment, stream, index,
                      operand_name, size) != expected:
        raise SchemaError("forward/DGRAD producer does not write the claimed expert operand",
                          path=f"moe_wgrad.producer[{action_ref}]")


def _derive(bridge, sequence):
    if (bridge.source_moe_sequence_ref != sequence.id
            or bridge.source_physical_case_ref == ""
            or len(bridge.paths) != 12):
        raise SchemaError("only twelve source-bound expert tiles on one public hardware case are enabled",
                          path="moe_wgrad.source")
    entries = []
    for path in bridge.paths:
        unit = _unit(sequence, path.layer)
        plan = unit.plan
        rank = path.expert
        m, n, k = (path.native_wgrad.m, path.native_wgrad.n,
                   path.native_wgrad.k)
        h, intermediate = unit.spec.hidden_size, unit.spec.intermediate_size
        source_fwd = next(action for action in plan.actions
                          if action.kind is MoeRectActionKind.EXPERT_FORWARD
                          and action.rank == rank)
        source_dgrad = next(action for action in plan.actions
                            if action.kind is MoeRectActionKind.EXPERT_DGRAD
                            and action.rank == rank)
        source_wgrad = next(action for action in plan.actions
                            if action.kind is MoeRectActionKind.EXPERT_WGRAD
                            and action.rank == rank)
        swiglu_backward, *_ = expert_dgrad_action_ids(plan.id,
                                                       source_dgrad.id)
        up_forward, swiglu_forward, down_forward = (
            expert_projection_action_ids(plan.id, source_fwd.id))
        up_wgrad, down_wgrad, *_ = expert_wgrad_action_ids(
            plan.id, source_wgrad.id)
        native_action = {"gate": source_wgrad.id,
                         "up": up_wgrad, "down": down_wgrad}[path.projection]
        if (k != len(source_wgrad.assignment_refs)
                or k != len(source_dgrad.assignment_refs)
                or k != len(source_fwd.assignment_refs)
                or ((m, n) != ((intermediate, h) if path.projection == "down"
                               else (h, intermediate)))):
            raise SchemaError("native expert tile geometry differs from real forward/DGRAD/WGRAD source actions",
                              path=f"moe_wgrad.layer{path.layer}.expert{rank}")
        manifest = unit.linked_manifest
        activation = _buffer_slice(manifest, rank, ".activation",
                                   DType.FP16, 0, 2 * k * h)
        activated = _buffer_slice(manifest, rank, ".expert_activated",
                                  DType.FP16, 0, 2 * k * intermediate)
        backward = _buffer_slice(manifest, rank, ".backward_gradient",
                                 DType.FP16, 0, 2 * k * h)
        gate_up = _buffer_slice(
            manifest, rank, ".dgrad_gate_up", DType.FP16,
            (0 if path.projection == "gate" else 2 * k * intermediate),
            2 * k * intermediate,
        ) if path.projection != "down" else backward
        gradient_ref = next(ref for ref in source_wgrad.state_refs
                            if next(state for state in plan.state_bindings
                                    if state.id == ref).role.value ==
                               "expert_gradient")
        output = _buffer_slice(
            manifest, rank, f".state.{gradient_ref}", DType.FP32,
            {"gate": 0, "up": 1, "down": 2}[path.projection] * 4 * h * intermediate,
            path.native_wgrad.gradient_bytes,
        )
        _producer_matches(manifest, rank, source_fwd.id,
                          RecordOpcode.MATMUL, "input_address",
                          activation, activation.size_bytes)
        _producer_matches(manifest, rank, down_forward,
                          RecordOpcode.MATMUL, "input_address",
                          activated, activated.size_bytes)
        _producer_matches(manifest, rank, swiglu_forward,
                          RecordOpcode.SWIGLU, "output_address",
                          activated, activated.size_bytes)
        _producer_matches(manifest, rank, source_dgrad.id,
                          RecordOpcode.MATMUL, "input_address",
                          backward, backward.size_bytes)
        _producer_matches(manifest, rank, swiglu_backward,
                          RecordOpcode.SWIGLU_BACKWARD_TIMING,
                          "output_address",
                          _buffer_slice(manifest, rank, ".dgrad_gate_up",
                                        DType.FP16, 0, 4 * k * intermediate),
                          4 * k * intermediate)
        entries.append(MoeExpertPhysicalWgradOperandSource(
            path.layer, rank, path.projection, source_fwd.id,
            source_dgrad.id, swiglu_backward, source_wgrad.id,
            native_action,
            activation if path.projection != "down" else activated,
            gate_up, output, m, n, k,
        ))
    return tuple(entries)


def build_moe_full_train_physical_wgrad_operand_sources(
    bridge: MoeFullTrainGradientSourceBridge,
    sequence: MoeCompileSequence,
) -> MoeFullTrainPhysicalWgradOperandSources:
    sequence.validate("moe_wgrad.production_source")
    result = MoeFullTrainPhysicalWgradOperandSources(
        sequence.id, bridge.source_ir0_ref, _derive(bridge, sequence))
    result.validate_against(bridge, sequence)
    return result


__all__ = ["MoeWgradPhysicalSlice",
           "MoeExpertPhysicalWgradOperandSource",
           "MoeFullTrainPhysicalWgradOperandSources",
           "build_moe_full_train_physical_wgrad_operand_sources"]
