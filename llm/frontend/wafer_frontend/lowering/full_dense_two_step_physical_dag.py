"""Witness every two-step Dense source action using its real native records."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, RecordOpcode
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_dense_gradient_requirements import DenseFullTrainRequirements
from ..schema.full_training_physical_dag import (
    FullTrainingPhysicalDAG, build_full_training_physical_dag,
)
from ..schema.ir0 import OpKind, ResidualWorkload, SwiGluWorkload
from ..schema.ir2 import SemanticTaskKind
from ..schema.persistent_state import PersistentStateAccess, StateKind
from ..schema.train_n6 import TrainLinkedProgram


def dense_two_step_native_opcode_contract(plan: FlexibleDenseTrainPlan) -> tuple[
    dict[str, RecordOpcode], dict[str, RecordOpcode],
]:
    """Derive derivative opcode families solely from validated forward nodes."""
    forward = {node.id: node for node in plan.forward_graph.nodes}
    backward = {}
    for node in forward.values():
        if node.kind in (OpKind.CE_FORWARD, OpKind.EMBEDDING):
            continue
        opcode = {
            OpKind.GEMM: RecordOpcode.GEMM_DX_TIMING,
            OpKind.NORM: RecordOpcode.RMSNORM_BACKWARD_TIMING,
            OpKind.ATTENTION: RecordOpcode.ATTENTION_BACKWARD_TIMING,
            OpKind.ROPE: RecordOpcode.ROPE_BACKWARD_TIMING,
        }.get(node.kind)
        if node.kind is OpKind.ELEMENTWISE:
            opcode = (RecordOpcode.SWIGLU_BACKWARD_TIMING
                      if type(node.workload) is SwiGluWorkload else
                      RecordOpcode.RESIDUAL_BACKWARD_TIMING
                      if type(node.workload) is ResidualWorkload else None)
        if opcode is None:
            raise SchemaError("untyped Dense derivative opcode family", path=node.id)
        backward[f"backward::{node.id}"] = opcode
    wgrad = {}
    for template in plan.parameter_templates:
        kinds = {forward[ref].kind for ref in template.forward_consumer_refs}
        if len(kinds) != 1:
            raise SchemaError("parameter derivative has ambiguous source", path=template.state_ref)
        opcode = {
            OpKind.GEMM: RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            OpKind.NORM: RecordOpcode.NORM_GAMMA_WGRAD_TIMING,
            OpKind.EMBEDDING: RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING,
        }.get(next(iter(kinds)))
        if opcode is None:
            raise SchemaError("parameter derivative has no native opcode", path=template.state_ref)
        wgrad[template.wgrad_ref] = opcode
    return backward, wgrad


def build_full_dense_two_step_physical_dag(
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
) -> FullTrainingPhysicalDAG:
    """Record one verifiable physical STORE0 -> LOAD1 edge per StateABI.

    The DP1 FP32 sync is the WGRAD output's own BufferABI.  No no-op or
    synthetic action is inserted into the production GlobalActionDAG.
    """
    program.validate("full_dense_two_step_program")
    requirements.validate_against(plan)
    if requirements.steps != 2 or len(program.source.replicas) != 1:
        raise SchemaError("initial two-step native witness requires one DP1 replica",
                          path="requirements")
    context = program.source.replicas[0].lowering_context
    global_dag = context.global_dag
    manifest: LinkedProgramManifest = program.manifest
    states = {state.id: state for state in context.ir1.persistent_state_manifest.declarations}
    old_states = {state.id: state for state in plan.forward_graph.persistent_states}
    old_by_tensor = {state.identity.tensor_ref: state.id for state in old_states.values()}
    source_state = {}
    for state in states.values():
        if (state.identity.kind is not StateKind.TRAINABLE_PARAMETER
                or state.access is not PersistentStateAccess.READ_WRITE
                or state.identity.tensor_ref not in old_by_tensor):
            raise SchemaError("physical source has an unknown trainable state",
                              path="context.ir1.persistent_state_manifest")
        old_id = old_by_tensor[state.identity.tensor_ref]
        old = old_states[old_id]
        if (state.identity.instance_ref != old.identity.instance_ref
                or state.identity.mesh_ref != old.identity.mesh_ref
                or state.identity.shard_index != old.identity.shard_index
                or state.identity.generation != old.identity.generation
                or state.shape != old.shape or state.dtype != old.dtype):
            raise SchemaError("trainable state changed source owner/shape/dtype",
                              path="context.ir1.persistent_state_manifest")
        source_state[state.id] = old_id
    if set(source_state.values()) != set(old_states):
        raise SchemaError("every source parameter needs one real trainable StateABI",
                          path="context.ir1.persistent_state_manifest")
    binding_to_old = {binding.id: source_state[binding.state_ref]
                      for binding in context.ir1.persistent_state_manifest.bindings}
    by_id = {node.id: node for node in context.ir1.nodes}
    operation_by_action: dict[str, tuple[str, str, int, int | None, str]] = {}
    stores: dict[str, object] = {}
    loads: dict[str, list[object]] = {}
    for action in global_dag.actions:
        origin = action.origin_ref
        member = action.member_id
        source_node = (by_id.get(member) if member is not None else
                       by_id.get(getattr(origin, "node_ref", None)))
        if source_node is None:
            raise SchemaError("native action lacks real source IR1 node", path=action.id)
        node_ref = source_node.id.removesuffix("__dp0")
        if "::step0" in node_ref:
            step = 0
        elif "::step1" in node_ref:
            step = 1
        else:
            raise SchemaError("native action lacks exact step namespace", path=action.id)
        canonical = node_ref.replace(f"::step{step}_backward", "_backward")
        canonical = canonical.replace(f"::step{step}", "")
        binding_refs = {use.hbm_binding_ref for use in action.state_uses}
        if len(binding_refs) > 1:
            raise SchemaError("physical state action crosses source parameters", path=action.id)
        state_ref = binding_to_old[next(iter(binding_refs))] if binding_refs else None
        if action.task_kind is SemanticTaskKind.DMA_OUT:
            if state_ref is None:
                raise SchemaError("STORE lacks source parameter StateABI", path=action.id)
            operation = f"store::{state_ref}::r0::step{step}"
            phase = "optimizer"
            stores[f"{state_ref}::{step}"] = action
        elif action.task_kind is SemanticTaskKind.DMA_IN:
            if state_ref is None:
                raise SchemaError("LOAD lacks source parameter StateABI", path=action.id)
            operation = f"load::{state_ref}::r0::step{step}"
            phase = "forward" if source_node.phase.value == "fwd" else "backward"
            loads.setdefault(f"{state_ref}::{step}", []).append(action)
        elif source_node.kind is OpKind.OPTIMIZER_UPDATE:
            weight = source_node.inputs[0].removesuffix("__dp0")
            state_ref = old_by_tensor.get(weight)
            if state_ref is None:
                raise SchemaError("SGD lacks exact source weight", path=action.id)
            operation = f"sgd::{state_ref}::r0::step{step}"
            phase = "optimizer"
        else:
            operation = canonical
            phase = ("loss" if source_node.kind in (
                OpKind.CE_FORWARD, OpKind.CE_BACKWARD) else
                "optimizer" if source_node.kind is OpKind.OPTIMIZER_UPDATE else
                "backward" if source_node.phase.value in ("dgrad", "wgrad")
                else "forward")
        tensor_ref = (old_states[state_ref].identity.tensor_ref
                      if state_ref is not None else canonical)
        layer = next((index for index in (0, 1)
                      if f".layer{index}." in tensor_ref), None)
        operation_by_action[action.id] = (action.id, operation, step, layer, phase)
    expected = {path.parameter_state_ref for path in requirements.paths}
    expected_versions = {f"{ref}::{step}" for ref in expected
                         for step in (0, 1)}
    if set(stores) != expected_versions or set(loads) != expected_versions:
        raise SchemaError(
            "two-step physical state coverage is incomplete: "
            f"source={len(expected)} stores={len(stores)} loads={len(loads)} "
            f"store_parameters={len(set(key.split('::')[0] for key in stores))} "
            f"load_parameters={len(set(key.split('::')[0] for key in loads))}",
            path="actions",
        )
    state_edges = []
    for old_ref in sorted(expected):
        store = stores.get(f"{old_ref}::0")
        read = loads.get(f"{old_ref}::1")
        if store is None or not read:
            raise SchemaError("STORE0 needs same-state LOAD1", path=old_ref)
        first = min(read, key=lambda action: action.core_order_index)
        if any(item.core_order_index <= store.core_order_index for item in read):
            raise SchemaError("step1 parameter read precedes its step0 STORE",
                              path=old_ref)
        state_edges.append((store.id, first.id))
    backward, wgrad = dense_two_step_native_opcode_contract(plan)
    required = {
        *backward, *wgrad,
        *(path.named_optimizer_op_ref for path in requirements.paths),
        *(path.named_store_op_ref for path in requirements.paths),
    }
    result = build_full_training_physical_dag(
        fragments=manifest.fragments, streams=manifest.core_streams,
        source_artifact_ids=tuple(sorted((context.ir1.id, global_dag.id))),
        operation_by_action=operation_by_action,
        transport_edges=(), state_version_edges=tuple(sorted(state_edges)),
        required_operation_ids=tuple(sorted(required)),
    )
    return result


__all__ = ["build_full_dense_two_step_physical_dag",
           "dense_two_step_native_opcode_contract"]
