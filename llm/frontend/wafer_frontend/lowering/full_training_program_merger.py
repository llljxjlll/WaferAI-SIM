"""One typed physical timeline for source-backed Dense/MoE training phases.

The caller provides actual native forward, CE, shared backward and layer MoE
leaf fragments, the complete per-core record order and a physical dependency
DAG.  This linker never creates graph compute or gradient placeholder records.
It produces a single production LinkedProgramManifest only after every source
carrier and full-model physical operation gate passes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Mapping

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressOperandBinding, CommandFragment, CoreRuntimeBinding,
    EmptyCoreAckPolicy, LinkedCoreStream,
    LinkedProgramManifest, LogicalStartEvent, ManifestInputDigest,
    ManifestInputKind, ProgramControlEnvelope, ProgramFailurePolicy,
    ProgramSymbolDefinition, RecordOpcode, SemanticOperandId, RuntimeSymbol,
    RuntimeSymbolDefinition, RuntimeSymbolKind, StateOperandBinding,
)
from ..schema.common import DType, stable_artifact_id
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import DenseFullTrainRequirements
from ..schema.full_moe_shared_train_requirements import FullMoeSharedTrainRequirements
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.action import FusionPlan, StandaloneCollectivePlan
from ..schema.global_action import GlobalActionDAG, LogicalCoreRef
from ..schema.ir1 import IR1
from ..schema.ir2 import IR2ProjectionResult, IntraDieScheduleSet
from ..schema.serde import canonical_digest
from .full_training_timeline_linker import require_physical_operation_coverage
from .full_dense_gradient_physical_gate import (
    require_full_dense_physical_gradient_paths,
)
from .full_moe_shared_train_source_gate import (
    require_moe_full_train_production_source,
)
from .moe_full_model_linker import _interfaces


_PASS = "full_training_physical_program_merger"
_SCHEMA = "wafer_frontend.full_training_physical_program_merger/v1alpha1"


def require_full_training_opcode_matrix(dag: FullTrainingPhysicalDAG) -> None:
    """Hard physical minimum for a 2-step, 2-layer native CE + MoE model."""
    dag.validate()
    by_step = defaultdict(Counter)
    by_layer = defaultdict(Counter)
    for action in dag.actions:
        for _, _, opcode in action.executable_records:
            by_step[action.step][opcode] += 1
            if action.layer is not None:
                by_layer[(action.step, action.layer)][opcode] += 1
    if set(by_step) != {0, 1} or set(by_layer) != {
        (step, layer) for step in (0, 1) for layer in (0, 1)
    }:
        raise SchemaError("full TRAIN must physically run two steps and two layers",
                          path="full_training_opcode_matrix")
    for step in (0, 1):
        current = by_step[step]
        required = {
            RecordOpcode.CROSS_ENTROPY_FORWARD: 1,
            RecordOpcode.CROSS_ENTROPY_BACKWARD: 1,
            RecordOpcode.SGD_UPDATE: 1,
            RecordOpcode.LSU_LOAD: 1,
            RecordOpcode.LSU_STORE: 1,
        }
        if any(current[opcode] < minimum for opcode, minimum in required.items()):
            raise SchemaError("step lacks native loss/backward or real SGD state IO",
                              path=f"full_training_opcode_matrix.step[{step}]")
        for layer in (0, 1):
            local = by_layer[(step, layer)]
            minimum = {
                RecordOpcode.ATTENTION_EXACT: 1,
                RecordOpcode.RMSNORM: 2,
                RecordOpcode.RESIDUAL: 2,
                RecordOpcode.SWIGLU: 1,
                RecordOpcode.SWIGLU_BACKWARD_TIMING: 1,
                RecordOpcode.DTE_SEND: 1,
                RecordOpcode.DTE_RECV: 1,
                RecordOpcode.SGD_UPDATE: 1,
            }
            if any(local[opcode] < count for opcode, count in minimum.items()):
                raise SchemaError("layer lacks true Dense attention/norm/residual, MoE backward, transport or update",
                                  path=f"full_training_opcode_matrix.step[{step}].layer[{layer}]")


def require_independent_ce_loss_gradient_seed(
    manifest: LinkedProgramManifest,
    dag: FullTrainingPhysicalDAG,
    *,
    seed_abi_by_step: Mapping[int, str],
) -> None:
    """Reject using forward loss value as backward dLoss.

    Structure alone does not prove a nonzero seed: the final ProgramIO/NpuSim
    runner must supply and audit actual per-row FP32 initialization.  This
    oracle refuses the earlier timing-only CE surrogate even when its native
    opcode and SRAM lifecycle pass validation.
    """
    manifest.validate("training_independent_loss_seed_source")
    dag.validate()
    if set(seed_abi_by_step) != {0, 1}:
        raise SchemaError("every step requires its own typed dLoss input",
                          path="seed_abi_by_step")
    fragments = {fragment.id: fragment for fragment in manifest.fragments}
    abis = {abi.id: abi for fragment in manifest.fragments for abi
            in fragment.buffer_abi}
    closures = {(binding.fragment_id, binding.logical_core,
                 binding.fragment_record_index, binding.operand_id): binding
                for binding in manifest.address_operand_bindings}
    for step in (0, 1):
        matches = defaultdict(list)
        for action in dag.actions:
            if action.step != step:
                continue
            for fragment_id, index, opcode in action.executable_records:
                if opcode in (RecordOpcode.CROSS_ENTROPY_FORWARD,
                              RecordOpcode.CROSS_ENTROPY_BACKWARD):
                    matches[opcode].append((action.logical_core, fragment_id, index))
        if any(len(matches[opcode]) != 1 for opcode in (
            RecordOpcode.CROSS_ENTROPY_FORWARD,
            RecordOpcode.CROSS_ENTROPY_BACKWARD,
        )):
            raise SchemaError("native CE forward/backward must each be physical once per step",
                              path=f"seed_abi_by_step[{step}]")

        def bound(executable, operand):
            core, fragment_id, index = executable
            closure = closures.get((fragment_id, core, index, operand))
            if closure is None or len(closure.buffer_abi_ids) != 1:
                raise SchemaError("CE loss/gradient lacks one actual SRAM closure",
                                  path=f"seed_abi_by_step[{step}]")
            return abis[closure.buffer_abi_ids[0]]

        loss = bound(matches[RecordOpcode.CROSS_ENTROPY_FORWARD][0],
                     SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        input_location = matches[RecordOpcode.CROSS_ENTROPY_BACKWARD][0]
        gradient = bound(input_location, SemanticOperandId.COMPUTE_AUX_ADDRESS)
        leaf = fragments[input_location[1]]
        local = next(item for item in leaf.core_streams
                     if item.logical_core == input_location[0])
        operands = {item.name: item.literal_value for item
                    in local.records[input_location[2]].operands
                    if item.literal_value is not None}
        rows = operands.get("rank_rows")
        if (type(rows) is not int or rows < 1 or
                gradient.id != seed_abi_by_step[step]
                or gradient.dtype is not DType.FP32
                or gradient.tensor_slice.shape != (rows,)
                or gradient.size_bytes < rows * 4 or
                gradient.storage_id == loss.storage_id
                or gradient.binding_id == loss.binding_id
                or gradient.value_id == loss.value_id):
            raise SchemaError("CE backward dLoss must be an independent per-row FP32 seed, not forward loss",
                              path=f"seed_abi_by_step[{step}]")


def link_source_backed_full_training_timeline(
    *,
    dag: FullTrainingPhysicalDAG,
    fragments: tuple[CommandFragment, ...],
    core_bindings: tuple[CoreRuntimeBinding, ...],
    core_streams: tuple[LinkedCoreStream, ...],
    runtime_definitions: tuple[RuntimeSymbolDefinition, ...],
    program_definitions: tuple[ProgramSymbolDefinition, ...],
    address_bindings: tuple[AddressOperandBinding, ...],
    state_bindings: tuple[StateOperandBinding, ...],
    source_ir1_id: str,
    source_projection_id: str,
    source_schedule_set_id: str,
    ir1: IR1,
    fusion_plans: tuple[FusionPlan, ...],
    standalone_plans: tuple[StandaloneCollectivePlan, ...],
    projection: IR2ProjectionResult,
    schedule_set: IntraDieScheduleSet,
    global_action_dag: GlobalActionDAG,
    upstream_input_digests: tuple[ManifestInputDigest, ...],
    required_operations: Mapping[str, Mapping[RecordOpcode, int]],
    seed_abi_by_step: Mapping[int, str],
    dense_plan: FlexibleDenseTrainPlan,
    dense_gradient_requirements: DenseFullTrainRequirements,
    required_backward_opcodes: Mapping[str, RecordOpcode],
    required_wgrad_opcodes: Mapping[str, RecordOpcode],
    moe_sequence: MoeCompileSequence | None = None,
    moe_source_requirements: FullMoeSharedTrainRequirements | None = None,
) -> LinkedProgramManifest:
    """Construct and validate one real full-training physical program.

    IR1, projection and schedule digests must represent actual source artifacts;
    The physical receipt is only an independently sourced operation oracle:
    GLOBAL_ACTION_DAG is the *real* production GlobalActionDAG signed from
    IR1/projection/schedule.  Production validate_against is mandatory before
    offering a manifest to the finalizer; no physical receipt may masquerade
    as that production source artifact.  Every fragment's declared DAG must
    equal the real GlobalActionDAG id.  Finalizer still checks SRAM lifecycle,
    HBM relocation, branch/transport wait and ProgramIO state after linking.
    """
    has_moe_fragments = any(fragment.producer_pass ==
                            "flexible_moe_production_lowering"
                            for fragment in fragments)
    if (has_moe_fragments != (moe_sequence is not None and
                              moe_source_requirements is not None)
            or (moe_sequence is None) != (moe_source_requirements is None)):
        raise SchemaError("MoE TRAIN physical leaves need independent source MLP replacement requirements",
                          path="moe_source_requirements")
    if moe_source_requirements is not None:
        moe_source_requirements.validate_against(
            dense_plan, dense_gradient_requirements, moe_sequence,
        )
    dag.validate_against(
        fragments, core_streams,
        required_operation_ids=tuple(sorted(required_operations)),
    )
    require_full_training_opcode_matrix(dag)
    ir1.validate("full_training_source_ir1")
    projection.validate_against(ir1, fusion_plans, standalone_plans)
    schedule_set.validate_against(projection, ir1)
    global_action_dag.validate_against(ir1, projection, schedule_set)
    if (source_ir1_id != ir1.id
            or source_projection_id != projection.id
            or source_schedule_set_id != schedule_set.id
            or global_action_dag.id not in dag.source_artifact_ids
            or {action.id for action in dag.actions} !=
               {action.id for action in global_action_dag.actions}):
        raise SchemaError("source IR1/projection/schedule/global DAG and physical actions must all agree",
                          path="full_training_source_lineage")
    if any(fragment.source_global_dag_id != global_action_dag.id
           for fragment in fragments):
        raise SchemaError("TRAIN leaf carrier does not originate in real GlobalActionDAG",
                          path="fragments.source_global_dag_id")
    for kind, expected_source in (
        (ManifestInputKind.IR1, ir1),
        (ManifestInputKind.IR2_PROJECTION, projection),
        (ManifestInputKind.SCHEDULE_SET, schedule_set),
    ):
        if not any(item.kind is kind
                   and item.artifact_id == expected_source.id
                   and item.schema_version == expected_source.schema_version
                   and item.digest == canonical_digest(expected_source)
                   for item in upstream_input_digests):
            raise SchemaError("IR1/projection/schedule requires actual exact input digest",
                              path=f"upstream_input_digests[{kind.value}]")
    if any(input_digest.kind in (ManifestInputKind.GLOBAL_ACTION_DAG,
                                 ManifestInputKind.COMMAND_FRAGMENT)
           for input_digest in upstream_input_digests):
        raise SchemaError("Global DAG and carriers must be signed by this merger",
                          path="upstream_input_digests")
    if tuple(sorted(core_bindings, key=lambda item:
                    (item.logical_core.die_id, item.logical_core.local_core_id))) != core_bindings:
        raise SchemaError("TRAIN physical core bindings must be canonical",
                          path="core_bindings")
    cores = tuple(binding.logical_core for binding in core_bindings)
    if cores != tuple(stream.logical_core for stream in core_streams):
        raise SchemaError("TRAIN stream must follow each real physical core",
                          path="core_streams")

    declared_runtime = {symbol.id: symbol for fragment in fragments
                        for symbol in fragment.runtime_symbols}
    declared_program = {symbol.id: symbol for fragment in fragments
                        for symbol in fragment.program_symbols}
    runtime = {}
    for definition in runtime_definitions:
        if definition.symbol.kind is RuntimeSymbolKind.START_TAG:
            raise SchemaError("source START_TAG must be replaced by one timeline START/core",
                              path="runtime_definitions")
        if definition.symbol.id not in declared_runtime or (
            definition.symbol.id in runtime and runtime[definition.symbol.id] != definition
        ):
            raise SchemaError("physical runtime symbol has collision or is unclaimed",
                              path=definition.symbol.id)
        runtime[definition.symbol.id] = definition
    if set(runtime) != set(declared_runtime) or any(
        runtime[symbol_id].symbol != symbol for symbol_id, symbol
        in declared_runtime.items()
    ):
        raise SchemaError("all source runtime symbols need one exact definition",
                          path="runtime_definitions")
    program = {}
    for definition in program_definitions:
        if definition.symbol.id not in declared_program or (
            definition.symbol.id in program and program[definition.symbol.id] != definition
        ):
            raise SchemaError("physical program symbol has collision or is unclaimed",
                              path=definition.symbol.id)
        program[definition.symbol.id] = definition
    if set(program) != set(declared_program) or any(
        program[symbol_id].symbol != symbol for symbol_id, symbol
        in declared_program.items()
    ):
        raise SchemaError("all source program symbols need one exact definition",
                          path="program_definitions")
    name_counts = Counter(item.name for item in program.values())
    program = {key: (replace(value, name=f"{value.name}.{key[-8:]}")
                     if name_counts[value.name] > 1 else value)
               for key, value in program.items()}

    starts = []
    for stream in core_streams:
        if not stream.records:
            raise SchemaError("full training must physically activate every core",
                              path=f"core_streams[{stream.logical_core}]")
        first = stream.records[0].source_global_action_id
        symbol = RuntimeSymbol(stable_artifact_id(
            "full_training_start", {"dag": dag.id,
                                    "core": stream.logical_core, "first": first},
            schema_version=_SCHEMA), RuntimeSymbolKind.START_TAG, first)
        runtime[symbol.id] = RuntimeSymbolDefinition(symbol, (stream.logical_core,),
                                                     None, None)
        starts.append(LogicalStartEvent(stream.logical_core, symbol.id, 1))
    inputs = tuple(sorted((
        *upstream_input_digests,
        ManifestInputDigest(ManifestInputKind.GLOBAL_ACTION_DAG,
                            global_action_dag.id, global_action_dag.schema_version,
                            canonical_digest(global_action_dag)),
        *(ManifestInputDigest(ManifestInputKind.COMMAND_FRAGMENT,
                              fragment.id, fragment.schema_version,
                              canonical_digest(fragment)) for fragment in fragments),
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    if len(set((item.kind, item.artifact_id) for item in inputs)) != len(inputs):
        raise SchemaError("source digest repeats an artifact/kind", path="input_digests")
    result = LinkedProgramManifest.create(
        producer_pass=_PASS, capabilities=0,
        source_ir1_id=source_ir1_id,
        source_projection_id=source_projection_id,
        source_schedule_set_id=source_schedule_set_id,
        source_global_dag_id=global_action_dag.id, input_digests=inputs,
        fragments=tuple(sorted(fragments, key=lambda item: item.id)),
        fragment_interfaces=_interfaces(fragments),
        core_bindings=core_bindings, core_streams=core_streams,
        runtime_symbol_definitions=tuple(sorted(runtime.values(), key=lambda item: item.symbol.id)),
        program_symbol_definitions=tuple(sorted(program.values(), key=lambda item: item.symbol.id)),
        address_operand_bindings=tuple(sorted(address_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=tuple(sorted(state_bindings, key=lambda item: (
            item.logical_core.die_id, item.logical_core.local_core_id,
            item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        core_groups=(),
        envelope=ProgramControlEnvelope(cores, tuple(starts), cores, cores, cores,
                                        EmptyCoreAckPolicy.INCLUDE_EMPTY,
                                        ProgramFailurePolicy.ABORT_ALL),
    )
    result.validate("source_backed_full_training_program")
    result.validate_against(
        ir1, fusion_plans, standalone_plans, projection,
        schedule_set, global_action_dag, result.fragments,
        "source_backed_full_training_program",
    )
    require_independent_ce_loss_gradient_seed(result, dag,
                                             seed_abi_by_step=seed_abi_by_step)
    require_physical_operation_coverage(
        result,
        operation_by_action={action.id: action.operation_ref for action
                             in dag.actions},
        required_by_operation=required_operations,
    )
    if has_moe_fragments:
        require_moe_full_train_production_source(
            result, dag, dense_plan, dense_gradient_requirements,
            moe_sequence, moe_source_requirements,
        )
    require_full_dense_physical_gradient_paths(
        result, dense_plan, dense_gradient_requirements, dag,
        required_backward_opcodes=required_backward_opcodes,
        required_wgrad_opcodes=required_wgrad_opcodes,
    )
    return result


__all__ = ["require_full_training_opcode_matrix",
           "require_independent_ce_loss_gradient_seed",
           "link_source_backed_full_training_timeline"]
