"""Signed physical operand cut for a bounded Dense activation checkpoint experiment.

This module only identifies the forward replay and backward consumer.  It does
not change a linked program, invent activation StateABI, or certify execution.
The native runtime must separately prove actual HBM stores/loads and replay.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation, BufferABI, LinkedProgramManifest, LinkedRecordRef,
    RecordOpcode, RelocatableRecord,
    SemanticOperandId, StateABI,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import LogicalCoreRef
from ..schema.persistent_state import StateKind
from ..schema.serde import canonical_digest


@dataclass(frozen=True, slots=True)
class CheckpointRecordOperand:
    record: LinkedRecordRef
    core: LogicalCoreRef
    operand: SemanticOperandId
    buffer: BufferABI
    linked_position: int


@dataclass(frozen=True, slots=True)
class DenseCheckpointPhysicalCut:
    id: str
    manifest_id: str
    manifest_digest: str
    source_ir1_id: str
    source_projection_id: str
    source_schedule_set_id: str
    source_global_dag_id: str
    replay_input: CheckpointRecordOperand
    replay_weight: CheckpointRecordOperand
    replay_output: CheckpointRecordOperand
    backward_input: CheckpointRecordOperand
    replay_parameter_state: StateABI
    persistent_parameter_states: tuple[StateABI, ...]
    no_checkpoint_saved_bytes: int
    checkpoint_saved_bytes: int
    highest_parameter_end_bytes: int

    def validate(self) -> None:
        if self.replay_output.buffer.size_bytes != self.no_checkpoint_saved_bytes:
            raise SchemaError("full saved activation extent drifted",
                              path="dense_checkpoint_physical_cut")
        if self.replay_input.buffer.size_bytes != self.checkpoint_saved_bytes:
            raise SchemaError("checkpoint input extent drifted",
                              path="dense_checkpoint_physical_cut")
        if self.checkpoint_saved_bytes >= self.no_checkpoint_saved_bytes:
            raise SchemaError("checkpoint does not reduce the physical HBM tape",
                              path="dense_checkpoint_physical_cut")
        if self.replay_output.buffer.value_id != self.backward_input.buffer.value_id:
            raise SchemaError("backward consumes a different activation value",
                              path="dense_checkpoint_physical_cut")
        if self.replay_parameter_state.id not in {state.id for state in self.persistent_parameter_states}:
            raise SchemaError("replay weight loses its physical StateABI home",
                              path="dense_checkpoint_physical_cut")
        if not self.persistent_parameter_states:
            raise SchemaError("replay has no signed physical parameter homes",
                              path="dense_checkpoint_physical_cut")


def _bound_operand(manifest: LinkedProgramManifest, core: LogicalCoreRef,
                   ref: LinkedRecordRef, position: int,
                   operand: SemanticOperandId) -> CheckpointRecordOperand:
    declarations = {abi.id: abi for fragment in manifest.fragments
                    for abi in fragment.buffer_abi}
    hits = [binding for binding in manifest.address_operand_bindings
            if binding.fragment_id == ref.fragment_id
            and binding.logical_core == core
            and binding.fragment_record_index == ref.fragment_record_index
            and binding.operand_id is operand]
    if len(hits) != 1 or len(hits[0].buffer_abi_ids) != 1:
        raise SchemaError("checkpoint record needs one exact BufferABI operand",
                          path=f"checkpoint_operand.{operand.name}")
    abi = declarations.get(hits[0].buffer_abi_ids[0])
    if abi is None or abi.logical_core != core or hits[0].tensor_slices != (abi.tensor_slice,):
        raise SchemaError("checkpoint operand loses physical BufferABI closure",
                          path=f"checkpoint_operand.{operand.name}")
    return CheckpointRecordOperand(ref, core, operand, abi, position)


def _opcode(manifest: LinkedProgramManifest, core: LogicalCoreRef,
            ref: LinkedRecordRef) -> RecordOpcode:
    fragment = next((f for f in manifest.fragments if f.id == ref.fragment_id), None)
    if fragment is None:
        raise SchemaError("checkpoint record fragment disappeared", path="fragment_id")
    local = next((s for s in fragment.core_streams if s.logical_core == core), None)
    if local is None or ref.fragment_record_index >= len(local.records):
        raise SchemaError("checkpoint record physical stream disappeared",
                          path="fragment_record_index")
    return local.records[ref.fragment_record_index].opcode


def _find(manifest: LinkedProgramManifest, *, action_id: str,
          opcode: RecordOpcode) -> tuple[LogicalCoreRef, LinkedRecordRef, int]:
    hits = [(stream.logical_core, ref, position)
            for stream in manifest.core_streams
            for position, ref in enumerate(stream.records)
            if ref.source_global_action_id == action_id
            and _opcode(manifest, stream.logical_core, ref) is opcode]
    if len(hits) != 1:
        raise SchemaError("checkpoint needs exactly one native source record",
                          path=f"checkpoint_record.{action_id}.{opcode.name}")
    return hits[0]


def derive_dense_checkpoint_physical_cut(
    manifest: LinkedProgramManifest, *,
    replay_action_id: str,
    backward_action_id: str,
    replay_weight_operand: SemanticOperandId = SemanticOperandId.COMPUTE_DATA_ADDRESS,
) -> DenseCheckpointPhysicalCut:
    """Bind one actual forward MATMUL replay to one native CE backward input.

    The same source-signature cut can be executed with or without a checkpoint.
    A caller must not infer runtime HBM traffic or recompute solely from this
    declaration.
    """
    manifest.validate("dense_checkpoint_source_manifest")
    core, forward, fpos = _find(manifest, action_id=replay_action_id,
                                opcode=RecordOpcode.MATMUL)
    bcore, backward, bpos = _find(manifest, action_id=backward_action_id,
                                  opcode=RecordOpcode.CROSS_ENTROPY_BACKWARD)
    if core != bcore or bpos <= fpos:
        raise SchemaError("backward must follow same-core replay producer",
                          path="checkpoint_record_order")
    src = _bound_operand(manifest, core, forward, fpos,
                         SemanticOperandId.COMPUTE_INPUT_ADDRESS)
    weight = _bound_operand(manifest, core, forward, fpos, replay_weight_operand)
    output = _bound_operand(manifest, core, forward, fpos,
                            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
    consumer = _bound_operand(manifest, bcore, backward, bpos,
                              SemanticOperandId.COMPUTE_INPUT_ADDRESS)
    a, b = output.buffer, consumer.buffer
    if (a.value_id, a.storage_id, a.region_ref, a.region_offset_bytes,
        a.tensor_slice, a.dtype, a.layout, a.size_bytes) != (
        b.value_id, b.storage_id, b.region_ref, b.region_offset_bytes,
        b.tensor_slice, b.dtype, b.layout, b.size_bytes):
        raise SchemaError("native backward operand differs from replay output",
                          path="checkpoint_backward_activation")
    if src.buffer.dtype != output.buffer.dtype or weight.buffer.dtype != output.buffer.dtype:
        raise SchemaError("replayed GEMM inputs are not the same physical dtype",
                          path="checkpoint_replay_dtype")
    states = tuple(sorted((state for fragment in manifest.fragments
                           for state in fragment.state_abi
                           if state.kind in (StateKind.PARAMETER,
                                             StateKind.TRAINABLE_PARAMETER)),
                          key=lambda s: (s.die_id, s.address, s.state_ref)))
    if len(states) != 15 or len({(s.state_ref, s.die_id) for s in states}) != 15:
        raise SchemaError("bounded L2 replay requires 15 signed parameter StateABI homes",
                          path="checkpoint_parameters")
    # Follow the exact LSU_LOAD HBM StateABI closure into the replay weight
    # staging BufferABI.  Merely signing all fifteen states is insufficient.
    state_by_id = {state.id: state for state in states}
    replay_homes = []
    for binding in manifest.state_operand_bindings:
        if binding.logical_core != core or binding.state_abi_id not in state_by_id:
            continue
        if _opcode(manifest, core, LinkedRecordRef(
                binding.fragment_id, binding.fragment_record_index,
                replay_action_id)) is not RecordOpcode.LSU_LOAD:
            continue
        destinations = [b for b in manifest.address_operand_bindings
                        if b.fragment_id == binding.fragment_id
                        and b.logical_core == core
                        and b.fragment_record_index == binding.fragment_record_index
                        and b.operand_id is SemanticOperandId.DESTINATION_ADDRESS]
        if len(destinations) != 1:
            continue
        abi_ids = destinations[0].buffer_abi_ids
        if weight.buffer.id in abi_ids:
            replay_homes.append(state_by_id[binding.state_abi_id])
    if len(replay_homes) != 1 or replay_homes[0].size_bytes != weight.buffer.size_bytes:
        raise SchemaError("LM-head replay weight lacks one exact LSU_LOAD StateABI",
                          path="checkpoint_replay_parameter")
    highest = max(state.address + state.size_bytes for state in states)
    semantic = {
        "manifest_id": manifest.id,
        "manifest_digest": canonical_digest(manifest),
        "replay_input": src,
        "replay_weight": weight,
        "replay_output": output,
        "backward_input": consumer,
        "parameter_state_ids": tuple(s.id for s in states),
        "replay_parameter_state_id": replay_homes[0].id,
        "source_ids": (manifest.source_ir1_id, manifest.source_projection_id,
                       manifest.source_schedule_set_id, manifest.source_global_dag_id),
    }
    cut = DenseCheckpointPhysicalCut(
        stable_artifact_id("dense_checkpoint_physical_cut", semantic,
                           schema_version="wafer_frontend.dense_checkpoint_physical_cut/v1alpha1"),
        manifest.id, semantic["manifest_digest"],
        manifest.source_ir1_id, manifest.source_projection_id,
        manifest.source_schedule_set_id, manifest.source_global_dag_id,
        src, weight, output, consumer, replay_homes[0], states,
        output.buffer.size_bytes, src.buffer.size_bytes, highest)
    cut.validate()
    return cut


@dataclass(frozen=True, slots=True)
class DenseCheckpointReplayTemplate:
    """Original native MATMUL bits to duplicate in a new signed fragment record."""

    id: str
    source_cut_id: str
    source_fragment_id: str
    source_fragment_record_index: int
    source_record_digest: str
    source_relocations_digest: str
    record: RelocatableRecord
    address_relocations: tuple[AddressRelocation, ...]

    def validate(self, cut: DenseCheckpointPhysicalCut) -> None:
        if (self.source_cut_id != cut.id
                or self.source_fragment_id != cut.replay_output.record.fragment_id
                or self.source_fragment_record_index !=
                    cut.replay_output.record.fragment_record_index
                or self.record.opcode is not RecordOpcode.MATMUL
                or self.record.source_global_action_id !=
                    cut.replay_output.record.source_global_action_id
                or canonical_digest(self.record) != self.source_record_digest
                or canonical_digest(self.address_relocations) !=
                    self.source_relocations_digest):
            raise SchemaError("replay template differs from original signed MATMUL",
                              path="dense_checkpoint_replay_template")
        expected = {
            SemanticOperandId.COMPUTE_INPUT_ADDRESS,
            SemanticOperandId.COMPUTE_DATA_ADDRESS,
            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
        }
        if {r.operand_id for r in self.address_relocations} != expected:
            raise SchemaError("replay template lost exact three MATMUL relocations",
                              path="dense_checkpoint_replay_template.address_relocations")


def derive_dense_checkpoint_replay_template(
    manifest: LinkedProgramManifest,
    cut: DenseCheckpointPhysicalCut,
) -> DenseCheckpointReplayTemplate:
    """Freeze exact source record/relocations; runtime must create a NEW record."""
    manifest.validate("checkpoint_replay_source")
    if manifest.id != cut.manifest_id or canonical_digest(manifest) != cut.manifest_digest:
        raise SchemaError("replay source manifest changed after cut signature",
                          path="dense_checkpoint_replay_template")
    ref, core = cut.replay_output.record, cut.replay_output.core
    fragment = next((f for f in manifest.fragments if f.id == ref.fragment_id), None)
    local = next((stream for stream in fragment.core_streams
                  if stream.logical_core == core), None) if fragment is not None else None
    if local is None or ref.fragment_record_index >= len(local.records):
        raise SchemaError("original replay native record disappeared",
                          path="dense_checkpoint_replay_template")
    record = local.records[ref.fragment_record_index]
    relocations = tuple(sorted(
        (r for r in local.address_relocations
         if r.record_index == ref.fragment_record_index),
        key=lambda r: int(r.operand_id)))
    semantic = {
        "source_cut_id": cut.id,
        "source_fragment_id": ref.fragment_id,
        "source_fragment_record_index": ref.fragment_record_index,
        "source_record_digest": canonical_digest(record),
        "source_relocations_digest": canonical_digest(relocations),
    }
    result = DenseCheckpointReplayTemplate(
        stable_artifact_id("dense_checkpoint_replay_template", semantic,
                           schema_version="wafer_frontend.dense_checkpoint_replay_template/v1alpha1"),
        **semantic, record=record, address_relocations=relocations)
    result.validate(cut)
    return result


@dataclass(frozen=True, slots=True)
class DenseCheckpointActivationTape:
    """Exact new activation slot, separate from the old aggregate P3 inventory."""

    id: str
    source_cut_id: str
    saved_buffer_abi_id: str
    saved_value_id: str
    hbm_binding_ref: str
    die_id: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    state_kind: str
    state_lifetime: str
    state_access: str
    activation_hbm_address: int
    size_bytes: int
    hbm_capacity_bytes: int
    checkpoint_enabled: bool

    def validate(self, cut: DenseCheckpointPhysicalCut) -> None:
        if self.source_cut_id != cut.id:
            raise SchemaError("activation tape is not bound to the linked source cut",
                              path="activation_tape.source_cut_id")
        expected_abi = (cut.replay_input.buffer.id if self.checkpoint_enabled
                        else cut.replay_output.buffer.id)
        expected_bytes = (cut.checkpoint_saved_bytes if self.checkpoint_enabled
                          else cut.no_checkpoint_saved_bytes)
        if (self.saved_buffer_abi_id != expected_abi or self.size_bytes != expected_bytes
                or self.saved_value_id != (cut.replay_input.buffer.value_id
                    if self.checkpoint_enabled else cut.replay_output.buffer.value_id)
                or self.hbm_binding_ref in {state.hbm_binding_ref
                    for state in cut.persistent_parameter_states}
                or self.die_id != cut.replay_parameter_state.die_id
                or self.state_kind != "activation"
                or self.state_lifetime != "step"
                or self.state_access != "read_write"):
            raise SchemaError("activation tape size/BufferABI differs from signed operand",
                              path="activation_tape")
        elements = 1
        for dim in self.shape:
            if dim <= 0:
                raise SchemaError("activation state shape must be positive",
                                  path="activation_tape.shape")
            elements *= dim
        dtype_bytes = {"fp16": 2, "fp32": 4, "int32": 4}.get(self.dtype)
        if dtype_bytes is None or elements * dtype_bytes != self.size_bytes:
            raise SchemaError("activation StateABI shape/dtype disagrees with physical bytes",
                              path="activation_tape.shape")
        if self.layout != (cut.replay_input.buffer.layout if self.checkpoint_enabled
                          else cut.replay_output.buffer.layout):
            raise SchemaError("activation state layout drifted from original BufferABI",
                              path="activation_tape.layout")
        if self.activation_hbm_address < cut.highest_parameter_end_bytes:
            raise SchemaError("activation tape overlaps signed parameter StateABI",
                              path="activation_tape.activation_hbm_address")
        if self.activation_hbm_address + self.size_bytes > self.hbm_capacity_bytes:
            raise SchemaError("activation tape exceeds the real HBM home capacity",
                              path="activation_tape.hbm_capacity_bytes")


def derive_dense_checkpoint_activation_tape(
    cut: DenseCheckpointPhysicalCut, *,
    checkpoint_enabled: bool,
    hbm_capacity_bytes: int,
    alignment_bytes: int = 64,
) -> DenseCheckpointActivationTape:
    """Declare one same-hardware HBM slot; native LSU must prove the transfers."""
    cut.validate()
    if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
        raise SchemaError("HBM activation alignment must be a power of two",
                          path="alignment_bytes")
    if hbm_capacity_bytes <= 0 or hbm_capacity_bytes % alignment_bytes:
        raise SchemaError("physical HBM home must follow channel granularity",
                          path="hbm_capacity_bytes")
    base = (cut.highest_parameter_end_bytes + alignment_bytes - 1) & -alignment_bytes
    source = (cut.replay_input.buffer if checkpoint_enabled
              else cut.replay_output.buffer)
    semantic = {
        "source_cut_id": cut.id,
        "saved_buffer_abi_id": source.id,
        "saved_value_id": source.value_id,
        "hbm_binding_ref": stable_artifact_id(
            "checkpoint_activation_hbm_binding",
            {"source_cut_id": cut.id, "saved_buffer_abi_id": source.id,
             "die_id": cut.replay_parameter_state.die_id},
            schema_version="wafer_frontend.checkpoint_activation_hbm_binding/v1alpha1"),
        "die_id": cut.replay_parameter_state.die_id,
        "shape": source.tensor_slice.shape,
        "dtype": source.dtype.value,
        "layout": source.layout,
        "state_kind": "activation",
        "state_lifetime": "step",
        "state_access": "read_write",
        "activation_hbm_address": base,
        "size_bytes": source.size_bytes,
        "hbm_capacity_bytes": hbm_capacity_bytes,
        "checkpoint_enabled": checkpoint_enabled,
    }
    tape = DenseCheckpointActivationTape(
        stable_artifact_id("dense_checkpoint_activation_tape", semantic,
                           schema_version="wafer_frontend.dense_checkpoint_activation_tape/v1alpha1"),
        **semantic)
    tape.validate(cut)
    return tape


__all__ = ["CheckpointRecordOperand", "DenseCheckpointPhysicalCut",
           "derive_dense_checkpoint_physical_cut",
           "DenseCheckpointReplayTemplate",
    "derive_dense_checkpoint_replay_template",
    "DenseCheckpointActivationTape",
           "derive_dense_checkpoint_activation_tape"]
