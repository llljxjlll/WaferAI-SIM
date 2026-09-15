"""Reusable typed phase cuts and HBM placement for one training timeline.

The helpers preserve real fragment records and persistent-state extents.
They do not synthesize loss, backward work, gradient contents, or runtime
status.  A caller must supply the production Dense forward/backward pieces
and construct a single LinkedProgramManifest before claiming E2E execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Mapping

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    BufferABI, LinkedCoreStream, LinkedProgramManifest, LinkedRecordRef,
    ProgramSymbolDefinition, RecordOpcode, SemanticOperandId, StateABI,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.flexible_dense_train import DenseTrainParameterTemplate
from ..schema.flexible_moe import (
    FlexibleMoeExecutablePlan, MoeRectActionKind, MoeRectFlowStage,
)
from ..schema.global_action import LogicalCoreRef
from ..schema.persistent_state import HbmAddressSpace, PersistentStateAccess, StateKind


_TRAIN_LINK_SCHEMA = "wafer_frontend.full_training_timeline_linker/v1alpha1"
_MOE_HBM_BASE = 1 << 24
_MOE_HBM_LAYER_STRIDE = 1 << 20


@dataclass(frozen=True, slots=True)
class PhysicalTrainingPhaseCut:
    """Exact ordered record refs on each side of a shared loss/backward edge."""

    source_manifest_id: str
    forward: tuple[LinkedCoreStream, ...]
    backward: tuple[LinkedCoreStream, ...]
    first_backward_index: tuple[tuple[LogicalCoreRef, int], ...]


@dataclass(frozen=True, slots=True)
class TrainingParameterHome:
    """One exact physical home shared by forward LOAD and backward SGD/STORE."""

    state_ref: str
    die_id: int
    forward_abi_id: str
    trainable_abi: StateABI


@dataclass(frozen=True, slots=True)
class ForwardCrossEntropyTape:
    """Original physical logits/labels/loss and their terminal free records."""

    logical_core: LogicalCoreRef
    forward_compute_ref: LinkedRecordRef
    logits: BufferABI
    labels: BufferABI
    per_row_loss: BufferABI
    terminal_frees: tuple[LinkedRecordRef, LinkedRecordRef, LinkedRecordRef]
    original_operand_definitions: tuple[
        ProgramSymbolDefinition, ProgramSymbolDefinition, ProgramSymbolDefinition
    ]
    terminal_free_core_positions: tuple[int, int, int]


def locate_forward_cross_entropy_tape(
    manifest: LinkedProgramManifest,
    *,
    action_id: str,
) -> ForwardCrossEntropyTape:
    """Find all three actual native CE inputs before moving their frees.

    Native CE backward consumes the original FP16 logits, INT32 labels and
    FP32 per-row forward loss as upstream; all three must remain allocated
    until the 0x1F record finishes.  A consumer must append the original three
    FREE records after its native backward, preserving their BufferABI closure.
    """
    manifest.validate("forward_cross_entropy_tape_source")
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    buffers = {abi.id: abi for fragment in manifest.fragments
               for abi in fragment.buffer_abi}
    bindings = {(b.fragment_id, b.logical_core,
                 b.fragment_record_index, b.operand_id): b
                for b in manifest.address_operand_bindings}
    definitions = {definition.symbol.id: definition
                   for definition in manifest.program_symbol_definitions}
    compute = []
    for stream in manifest.core_streams:
        for index, ref in enumerate(stream.records):
            if ref.source_global_action_id != action_id:
                continue
            source = next((local for local in fragments[ref.fragment_id].core_streams
                           if local.logical_core == stream.logical_core), None)
            if source is None:
                raise SchemaError("native CE tape lacks its physical core stream",
                                  path="forward_cross_entropy_tape")
            if source.records[ref.fragment_record_index].opcode is RecordOpcode.CROSS_ENTROPY_FORWARD:
                compute.append((stream, index, ref))
    if len(compute) != 1:
        raise SchemaError("forward tape requires exactly one native CE producer",
                          path="forward_cross_entropy_tape")
    stream, position, ref = compute[0]
    endpoints, operand_definitions = [], []
    source = next(local for local in fragments[ref.fragment_id].core_streams
                  if local.logical_core == stream.logical_core)
    compute_record = source.records[ref.fragment_record_index]
    for operand, dtype in (
        (SemanticOperandId.COMPUTE_INPUT_ADDRESS, DType.FP16),
        (SemanticOperandId.COMPUTE_DATA_ADDRESS, DType.INT32),
        (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, DType.FP32),
    ):
        binding = bindings.get((ref.fragment_id, stream.logical_core,
                                ref.fragment_record_index, operand))
        if binding is None or len(binding.buffer_abi_ids) != 1:
            raise SchemaError("CE tape operand lacks one physical BufferABI",
                              path=f"forward_cross_entropy_tape[{operand.name}]")
        abi = buffers[binding.buffer_abi_ids[0]]
        if abi.dtype is not dtype or abi.size_bytes == 0:
            raise SchemaError("native CE tape dtype/size disagrees with opcode",
                              path=f"forward_cross_entropy_tape[{operand.name}]")
        endpoints.append(abi)
        operands = [item for item in compute_record.operands
                    if item.operand_id is operand and item.symbol_ref in definitions]
        if len(operands) != 1:
            raise SchemaError("native CE operand lacks its original ProgramSymbolDefinition",
                              path=f"forward_cross_entropy_tape[{operand.name}]")
        operand_definitions.append(definitions[operands[0].symbol_ref])
    if len({abi.storage_id for abi in endpoints}) != 3:
        raise SchemaError("CE logits/labels/loss must own disjoint SRAM storage",
                          path="forward_cross_entropy_tape")
    frees, free_positions = [], []
    for abi in endpoints:
        matches = []
        for free_position in range(position + 1, len(stream.records)):
            free_ref = stream.records[free_position]
            if free_ref.source_global_action_id != action_id:
                continue
            source = next(local for local in fragments[free_ref.fragment_id].core_streams
                          if local.logical_core == stream.logical_core)
            if source.records[free_ref.fragment_record_index].opcode is not RecordOpcode.SRAM_FREE:
                continue
            binding = bindings.get((free_ref.fragment_id, stream.logical_core,
                                    free_ref.fragment_record_index, SemanticOperandId.SYMBOL))
            if binding is not None and binding.buffer_abi_ids == (abi.id,):
                matches.append((free_ref, free_position))
        if len(matches) != 1:
            raise SchemaError("each CE tape storage needs one original late FREE",
                              path=f"forward_cross_entropy_tape[{abi.value_id}]")
        frees.append(matches[0][0])
        free_positions.append(matches[0][1])
    return ForwardCrossEntropyTape(stream.logical_core, ref, *endpoints,
                                   tuple(frees), tuple(operand_definitions),
                                   tuple(free_positions))


def reconcile_training_parameter_homes(
    forward: LinkedProgramManifest,
    backward: LinkedProgramManifest,
    templates: tuple[DenseTrainParameterTemplate, ...],
) -> tuple[TrainingParameterHome, ...]:
    """Anchor parameter read/write state to the same model and physical span.

    The forward leaf is usually PARAMETER/read_only and the training leaf is
    TRAINABLE_PARAMETER/read_write.  A merger must use each returned trainable
    StateABI for both directions, re-sign forward HBM symbols/closures, and
    separately prove the step0→step1 version transition; simple concatenation
    of the two distinct StateABI identities does not establish continuity.
    """
    forward.validate("training_forward_state_source")
    backward.validate("training_backward_state_source")
    by_key = lambda source: {
        (abi.state_ref, abi.die_id): abi
        for fragment in source.fragments for abi in fragment.state_abi
    }
    reads, writes = by_key(forward), by_key(backward)
    expected = {(template.state_ref, die_id): template
                for template in templates for die_id in template.owner_ranks}
    if (len(reads) != len(expected) or len(writes) != len(expected)
            or set(reads) != set(expected) or set(writes) != set(expected)):
        raise SchemaError("forward/backward model parameter owners must be bijective",
                          path="training_parameter_homes")

    def closure_counts(source: LinkedProgramManifest) -> Counter:
        fragments = {fragment.id: fragment for fragment in source.fragments}
        result = Counter()
        for binding in source.state_operand_bindings:
            fragment = fragments[binding.fragment_id]
            stream = next((item for item in fragment.core_streams
                           if item.logical_core == binding.logical_core), None)
            if stream is None:
                raise SchemaError("state closure has no source physical stream",
                                  path="training_parameter_homes")
            abi = next((item for item in fragment.state_abi
                        if item.id == binding.state_abi_id), None)
            if abi is None:
                raise SchemaError("state closure lacks its source physical StateABI",
                                  path="training_parameter_homes")
            record = stream.records[binding.fragment_record_index]
            result[(abi.state_ref, abi.die_id, record.opcode)] += 1
        return result

    f_counts, b_counts = closure_counts(forward), closure_counts(backward)
    homes = []
    for key, template in sorted(expected.items()):
        f, b = reads[key], writes[key]
        if (f.kind is not StateKind.PARAMETER
                or f.access is not PersistentStateAccess.READ_ONLY
                or b.kind is not StateKind.TRAINABLE_PARAMETER
                or b.access is not PersistentStateAccess.READ_WRITE
                or (f.address, f.size_bytes, f.shape, f.dtype, f.layout,
                    f.alignment_bytes) !=
                   (b.address, b.size_bytes, b.shape, b.dtype, b.layout,
                    b.alignment_bytes)
                or f.size_bytes != template.weight_bytes
                or f_counts[(key[0], key[1], RecordOpcode.LSU_LOAD)] != 1
                or b_counts[(key[0], key[1], RecordOpcode.LSU_LOAD)] != 1
                or b_counts[(key[0], key[1], RecordOpcode.LSU_STORE)] != 1):
            raise SchemaError("parameter forward read and SGD store lack one exact HBM home",
                              path=f"training_parameter_homes[{key}]")
        homes.append(TrainingParameterHome(key[0], key[1], f.id, b))
    return tuple(homes)


def require_physical_operation_coverage(
    manifest: LinkedProgramManifest,
    *,
    operation_by_action: Mapping[str, str],
    required_by_operation: Mapping[str, Mapping[RecordOpcode, int]],
) -> None:
    """Reject missing physical work for any named step/layer graph operation.

    The caller tags each independent step and layer operation separately.
    A plan action or a MATMUL placeholder for NORM/CE cannot satisfy another
    graph operation.  This oracle does not assert functional numeric fidelity.
    """
    manifest.validate("training_physical_coverage_source")
    if not operation_by_action or not required_by_operation:
        raise SchemaError("full training needs named actions and required physical work",
                          path="required_by_operation")
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    actual: dict[str, Counter] = {op: Counter() for op in required_by_operation}
    for stream in manifest.core_streams:
        for ref in stream.records:
            op = operation_by_action.get(ref.source_global_action_id)
            if op not in actual:
                continue
            fragment = fragments[ref.fragment_id]
            local = next((item for item in fragment.core_streams
                          if item.logical_core == stream.logical_core), None)
            if local is None:
                raise SchemaError("physical graph action missing its local stream",
                                  path=f"operation[{op}]")
            actual[op][local.records[ref.fragment_record_index].opcode] += 1
    for op, requirements in required_by_operation.items():
        if not requirements or any(
            type(opcode) is not RecordOpcode or type(count) is not int or count < 1
            for opcode, count in requirements.items()
        ):
            raise SchemaError("physical operator oracle needs positive typed opcode counts",
                              path=f"operation[{op}]")
        missing = {opcode.name: count - actual[op][opcode]
                   for opcode, count in requirements.items()
                   if actual[op][opcode] < count}
        if missing:
            raise SchemaError(f"graph operation lacks physical work: {missing}",
                              path=f"operation[{op}]")


def cut_physical_training_phases(
    manifest: LinkedProgramManifest,
    first_backward_action_by_core: Mapping[LogicalCoreRef, tuple[str, RecordOpcode]],
    *,
    forbid_early_sram_free: bool = False,
    forbid_late_sram_alloc: bool = False,
) -> PhysicalTrainingPhaseCut:
    """Slice original physical references once; reject misplaced lifecycles.

    All fragment references remain the original linked references.  Callers
    can rebase their fragments and concatenate these streams around real CE
    loss/backward records without changing operation bytes or work.
    """
    manifest.validate("physical_training_phase_source")
    if not first_backward_action_by_core or set(first_backward_action_by_core) != {
        stream.logical_core for stream in manifest.core_streams
    }:
        raise SchemaError("training phase cut must cover every physical core exactly",
                          path="first_backward_action_by_core")
    forward, backward, indices = [], [], []
    for stream in manifest.core_streams:
        action_id, opcode = first_backward_action_by_core[stream.logical_core]
        local = {
            fragment.id: next((item for item in fragment.core_streams
                               if item.logical_core == stream.logical_core), None)
            for fragment in manifest.fragments
        }
        def actual(ref: LinkedRecordRef):
            source = local[ref.fragment_id]
            if source is None:
                raise SchemaError("training phase record is absent from its local core",
                                  path=ref.fragment_id)
            return source.records[ref.fragment_record_index]
        cuts = [
            index for index, ref in enumerate(stream.records)
            if ref.source_global_action_id == action_id and actual(ref).opcode is opcode
        ]
        if len(cuts) != 1 or cuts[0] in (0, len(stream.records) - 1):
            raise SchemaError("training phase needs one physical loss/backward boundary",
                              path=f"core[{stream.logical_core}]")
        cut = cuts[0]
        for index, ref in enumerate(stream.records):
            record = actual(ref)
            if index < cut and record.opcode in (
                (RecordOpcode.LSU_STORE, RecordOpcode.SGD_UPDATE)
                + ((RecordOpcode.SRAM_FREE,) if forbid_early_sram_free else ())
            ):
                raise SchemaError("training tape/state freed or updated before loss",
                                  path=f"core[{stream.logical_core}].record[{index}]")
            if (forbid_late_sram_alloc and index >= cut
                    and record.opcode is RecordOpcode.SRAM_ALLOC_AT):
                raise SchemaError("training backing alloc moved after its forward tape",
                                  path=f"core[{stream.logical_core}].record[{index}]")
        forward.append(LinkedCoreStream(stream.logical_core, stream.runtime_core_id,
                                        stream.records[:cut]))
        backward.append(LinkedCoreStream(stream.logical_core, stream.runtime_core_id,
                                         stream.records[cut:]))
        indices.append((stream.logical_core, cut))
    return PhysicalTrainingPhaseCut(
        manifest.id, tuple(forward), tuple(backward), tuple(indices))


def cut_moe_training_unit(
    plan: FlexibleMoeExecutablePlan,
    manifest: LinkedProgramManifest,
) -> PhysicalTrainingPhaseCut:
    """Cut per-layer MoE after combine and before physical gradient dispatch."""
    if manifest.source_global_dag_id != plan.id:
        raise SchemaError("MoE phase source plan does not match production manifest",
                          path="manifest.source_global_dag_id")
    flows = [flow for flow in plan.flows
             if flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT
             and flow.source_rank == 0 and flow.destination_rank == 1]
    if len(flows) != 1:
        raise SchemaError("training MoE needs one rank0→rank1 backward gradient flow",
                          path="plan.flows")
    flow = flows[0]
    kinds = {
        MoeRectActionKind.SEND: (LogicalCoreRef(0, 0), RecordOpcode.DTE_SEND),
        MoeRectActionKind.RECV: (LogicalCoreRef(1, 0), RecordOpcode.DTE_RECV),
    }
    markers = {}
    for kind, (core, opcode) in kinds.items():
        matches = [action for action in plan.actions
                   if action.kind is kind and action.flow_ref == flow.id
                   and action.rank == core.die_id]
        if len(matches) != 1:
            raise SchemaError("MoE gradient boundary lacks one P2 transport",
                              path=f"plan.flows[{flow.id}]")
        markers[core] = (matches[0].id, opcode)
    return cut_physical_training_phases(
        manifest, markers, forbid_early_sram_free=True,
        forbid_late_sram_alloc=True,
    )


def rebase_train_state_home_ranges(
    manifest: LinkedProgramManifest,
    *,
    namespace: str,
    layer: int,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> dict[str, StateABI]:
    """Put one unchanged MoE expert/gate tensor per layer into finite global HBM.

    The same layer produces the same physical addresses at both SGD steps;
    namespace changes only typed StateABI identities.  No step state/version
    handoff is claimed until the real two-step ProgramIO and NpuSim run.
    """
    manifest.validate("training_state_rebase_source")
    if not namespace or layer < 0 or layer > 1 or len(hbm_address_spaces) != 2:
        raise SchemaError("training MoE state rebase requires two finite Die homes",
                          path="hbm_address_spaces")
    homes = {space.die_id: space for space in hbm_address_spaces}
    if set(homes) != {0, 1}:
        raise SchemaError("MoE training needs a rank0 and rank1 global HBM home",
                          path="hbm_address_spaces")
    for space in homes.values():
        space.validate("training_state_home_range")
    old = {abi.id: abi for fragment in manifest.fragments for abi in fragment.state_abi}
    rebased = {}
    for abi in old.values():
        home = homes[abi.die_id]
        address = (home.base_address + _MOE_HBM_BASE
                   + layer * _MOE_HBM_LAYER_STRIDE + abi.address)
        if (abi.address >= _MOE_HBM_LAYER_STRIDE
                or abi.size_bytes > _MOE_HBM_LAYER_STRIDE - abi.address
                or address + abi.size_bytes > home.base_address + home.size_bytes):
            raise SchemaError("MoE layer expert/router tensor exceeds finite HBM home",
                              path=abi.id)
        result = StateABI.create(
            state_ref=stable_artifact_id(
                "full_training_state", {"namespace": namespace,
                                        "layer": layer, "old": abi.state_ref},
                schema_version=_TRAIN_LINK_SCHEMA),
            hbm_binding_ref=stable_artifact_id(
                "full_training_state_binding",
                {"namespace": namespace, "layer": layer,
                 "old": abi.hbm_binding_ref},
                schema_version=_TRAIN_LINK_SCHEMA),
            kind=abi.kind, lifetime=abi.lifetime, access=abi.access,
            shape=abi.shape, dtype=abi.dtype, layout=abi.layout,
            die_id=abi.die_id, address=address, size_bytes=abi.size_bytes,
            alignment_bytes=abi.alignment_bytes,
        )
        result.validate(f"training_state[{abi.id}]")
        rebased[abi.id] = result
    return rebased


__all__ = [
    "PhysicalTrainingPhaseCut", "TrainingParameterHome", "ForwardCrossEntropyTape",
    "locate_forward_cross_entropy_tape",
    "reconcile_training_parameter_homes", "require_physical_operation_coverage",
    "cut_physical_training_phases",
    "cut_moe_training_unit", "rebase_train_state_home_ranges",
]
