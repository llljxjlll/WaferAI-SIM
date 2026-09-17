"""Prove complete Dense AdamW gradients and 75 HBM versions on native records."""
from __future__ import annotations

from collections import defaultdict
from math import prod

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.common import DType
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.full_training_physical_dag import (
    FullTrainingPhysicalDAG, build_full_training_physical_dag,
)
from ..schema.ir0 import AdamwUpdateWorkload, OpKind, StateAccessMode
from ..schema.ir2 import SemanticTaskKind, StateIoOrigin, StateUseAccess
from ..schema.n6 import _leaf_fragments
from ..schema.persistent_state import StateKind
from ..schema.train_n6 import TrainLinkedProgram
from .full_dense_two_step_physical_dag import dense_two_step_native_opcode_contract


def build_full_dense_adamw_physical_dag(
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
) -> FullTrainingPhysicalDAG:
    """Require WGRAD→AdamW and same-core STORE0→LOAD1 for every real state.

    This is a timing/physical dependency proof.  It does not assert numerical
    convergence, external-memory offload, or a particular optimizer payload.
    """
    program.validate("full_dense_adamw_program")
    plan.validate("full_dense_adamw_plan")
    if (plan.spec.tp_degree != 1 or plan.spec.dp_degree != 1
            or len(program.source.replicas) != 1):
        raise SchemaError("full AdamW physical gate currently needs TP1/DP1",
                          path="plan.spec")
    context = program.source.replicas[0].lowering_context
    graph = context.ir1
    manifest = program.manifest
    nodes = {node.id: node for node in graph.nodes}
    actions = {action.id: action for action in context.global_dag.actions}
    if len(nodes) != 170 or len(actions) != 520 or len(graph.state_accesses) != 200:
        raise SchemaError("two-step AdamW must retain complete source/N5 actions",
                          path="source.lowering_context")
    declarations = {state.id: state for state in
                    graph.persistent_state_manifest.declarations}
    bindings = {item.state_ref: item.id for item in
                graph.persistent_state_manifest.bindings}
    if len(declarations) != 75 or set(bindings) != set(declarations):
        raise SchemaError("AdamW needs seventy-five distinct physical StateABIs",
                          path="graph.persistent_state_manifest")
    value_dtype = {value.id: value.dtype for value in graph.values}
    access_by_node = defaultdict(list)
    for access in graph.state_accesses:
        access_by_node[access.node_ref].append(access)
    by_member = defaultdict(list)
    state_action = defaultdict(list)
    for action in actions.values():
        by_member[action.member_id].append(action)
        if isinstance(action.origin_ref, StateIoOrigin):
            state_action[action.origin_ref.state_access_ref].append(action)
    leaves = _leaf_fragments(manifest.fragments)
    leaf_by_id = {fragment.id: fragment for fragment in leaves}
    records = defaultdict(list)
    for stream in manifest.core_streams:
        for position, ref in enumerate(stream.records):
            fragment = leaf_by_id[ref.fragment_id]
            carrier = next((item for item in fragment.core_streams
                            if item.logical_core == stream.logical_core), None)
            if carrier is None:
                raise SchemaError("linked AdamW record has no source core stream",
                                  path=ref.fragment_id)
            record = carrier.records[ref.fragment_record_index]
            if record.source_global_action_id != ref.source_global_action_id:
                raise SchemaError("linked AdamW record changed source action",
                                  path=ref.fragment_id)
            records[ref.source_global_action_id].append(
                (stream.logical_core, position, record.opcode))
    if set(records) != set(actions):
        raise SchemaError("every AdamW action needs native record provenance",
                          path="manifest.core_streams")
    dependencies = {action.id: set(action.deps) for action in actions.values()}

    def reaches(predecessor: str, successor: str) -> bool:
        visited = set()
        stack = [successor]
        while stack:
            current = stack.pop()
            if current == predecessor:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(dependencies[current])
        return False

    backward, wgrad_opcodes = dense_two_step_native_opcode_contract(plan)
    if len(backward) != 24 or len(wgrad_opcodes) != 15:
        raise SchemaError("AdamW needs all two-layer typed derivative opcode families",
                          path="plan.forward_graph")
    version = {}
    state_edges = []
    gradient_paths = 0
    state_triplets = 0
    for template in plan.parameter_templates:
        weight = plan.forward_graph.persistent_states
        weight_ref = next(state.identity.tensor_ref for state in weight
                          if state.id == template.state_ref)
        if weight_ref is None:
            raise SchemaError("AdamW template has no source weight", path=template.state_ref)
        for step in (0, 1):
            suffix = f"::step{step}__dp0"
            wgrad_ref = f"{template.wgrad_ref}{suffix}"
            update_ref = f"adamw_update::{weight_ref}::tp0{suffix}"
            derivative = nodes.get(wgrad_ref)
            update = nodes.get(update_ref)
            if (derivative is None or update is None
                    or value_dtype[derivative.outputs[0]] is not DType.FP32
                    or type(update.workload) is not AdamwUpdateWorkload
                    or update.workload.step != step + 1):
                raise SchemaError("AdamW requires true FP32 WGRAD and typed step update",
                                  path=update_ref)
            source_wgrad = tuple(action for action in by_member[wgrad_ref]
                                 if action.task_kind is SemanticTaskKind.COMP)
            optimizer = tuple(action for action in by_member[update_ref]
                              if action.task_kind is SemanticTaskKind.COMP)
            if len(source_wgrad) != 1 or len(optimizer) != 1:
                raise SchemaError("one WGRAD and one AdamW compute must execute per parameter/step",
                                  path=update_ref)
            source_wgrad, optimizer = source_wgrad[0], optimizer[0]
            expected_wgrad = wgrad_opcodes[template.wgrad_ref]
            if (sum(opcode is expected_wgrad for _core, _position, opcode
                    in records[source_wgrad.id]) != 1
                    or sum(opcode is RecordOpcode.ADAMW_UPDATE
                           for _core, _position, opcode in records[optimizer.id]) != 1
                    or not reaches(source_wgrad.id, optimizer.id)):
                raise SchemaError("real WGRAD must reach its native AdamW update",
                                  path=update_ref)
            own = tuple(access for access in access_by_node[update_ref]
                        if access.mode is StateAccessMode.READ_WRITE)
            kinds = {declarations[access.state_ref].identity.kind for access in own}
            if len(own) != 5 or kinds != {
                StateKind.TRAINABLE_PARAMETER, StateKind.OPTIMIZER_MASTER,
                StateKind.OPTIMIZER_MOMENT1, StateKind.OPTIMIZER_MOMENT2,
                StateKind.OPTIMIZER_STEP,
            }:
                raise SchemaError("AdamW update lacks one weight/master/m/v/step state",
                                  path=update_ref)
            for access in own:
                state = declarations[access.state_ref]
                if state.dtype is (DType.INT32 if state.identity.kind is StateKind.OPTIMIZER_STEP
                                   else DType.FP16 if state.identity.kind is StateKind.TRAINABLE_PARAMETER
                                   else DType.FP32):
                    pass
                else:
                    raise SchemaError("AdamW physical state dtype changed", path=state.id)
                if prod(state.shape) == 0:
                    raise SchemaError("AdamW physical state has no payload", path=state.id)
                pair = state_action[access.id]
                loads = tuple(action for action in pair
                              if action.task_kind is SemanticTaskKind.DMA_IN)
                stores = tuple(action for action in pair
                               if action.task_kind is SemanticTaskKind.DMA_OUT)
                if len(loads) != 1 or len(stores) != 1:
                    raise SchemaError("AdamW state needs one real HBM LOAD and STORE",
                                      path=access.id)
                load, store = loads[0], stores[0]
                binding = bindings[access.state_ref]
                if (len(load.state_uses) != 1 or len(store.state_uses) != 1
                        or load.state_uses[0].hbm_binding_ref != binding
                        or store.state_uses[0].hbm_binding_ref != binding
                        or load.state_uses[0].access is not StateUseAccess.READ
                        or store.state_uses[0].access is not StateUseAccess.WRITE
                        or not reaches(load.id, optimizer.id)
                        or not reaches(optimizer.id, store.id)):
                    raise SchemaError("AdamW state LOAD→update→STORE lacks exact source binding",
                                      path=access.id)
                load_record = tuple((core, position) for core, position, opcode
                                    in records[load.id]
                                    if opcode is RecordOpcode.LSU_LOAD)
                store_record = tuple((core, position) for core, position, opcode
                                     in records[store.id]
                                     if opcode is RecordOpcode.LSU_STORE)
                if (len(load_record) != 1 or len(store_record) != 1
                        or load_record[0][0] != store_record[0][0]
                        or load_record[0][1] >= store_record[0][1]):
                    raise SchemaError("AdamW state native LSU LOAD/STORE order is wrong",
                                      path=access.id)
                key = state.id
                if step == 0:
                    version[key] = (store.id, store_record[0])
                else:
                    previous = version.get(key)
                    if (previous is None or previous[1][0] != load_record[0][0]
                            or previous[1][1] >= load_record[0][1]):
                        raise SchemaError(
                            f"AdamW STORE0 must precede same-state LOAD1: "
                            f"store={previous!r} load={(load.id, load_record[0])!r} "
                            f"direct={previous[0] in dependencies[load.id]}",
                            path=state.id)
                    state_edges.append((previous[0], load.id))
                state_triplets += 1
            gradient_paths += 1
    if (gradient_paths != 30 or state_triplets != 150
            or len(version) != 75 or len(state_edges) != 75):
        raise SchemaError("AdamW physical gradient/state version inventory is incomplete",
                          path="source")
    operation = {}
    for action in actions.values():
        origin = action.origin_ref
        ref = action.member_id or getattr(origin, "node_ref", None)
        node = nodes.get(ref)
        if node is None or action.logical_core is None:
            raise SchemaError("AdamW native action lacks one source node/core",
                              path=action.id)
        step = 0 if "::step0" in ref else 1 if "::step1" in ref else None
        if step is None:
            raise SchemaError("AdamW action lacks source step", path=action.id)
        layer = next((index for index in (0, 1)
                      if f".layer{index}." in ref), None)
        phase = ("loss" if node.kind in (OpKind.CE_FORWARD, OpKind.CE_BACKWARD)
                 else "optimizer" if node.kind is OpKind.OPTIMIZER_UPDATE
                 else "backward" if node.phase.value in ("dgrad", "wgrad")
                 else "forward")
        operation[action.id] = (action.id, ref, step, layer, phase)
    return build_full_training_physical_dag(
        fragments=leaves, streams=manifest.core_streams,
        source_artifact_ids=(graph.id, context.global_dag.id),
        operation_by_action=operation,
        transport_edges=(), state_version_edges=tuple(sorted(state_edges)),
        required_operation_ids=tuple(sorted(nodes)),
    )


def require_full_dense_adamw_physical_gradient_paths(
    candidate: FullTrainingPhysicalDAG,
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
) -> None:
    """Audit a stored receipt against independently rebuilt native source facts."""
    candidate.validate("full_dense_adamw_candidate")
    expected = build_full_dense_adamw_physical_dag(program, plan)
    if candidate != expected:
        raise SchemaError("AdamW physical gradient or StateABI version receipt differs from native source",
                          path="full_dense_adamw_candidate")


__all__ = [
    "build_full_dense_adamw_physical_dag",
    "require_full_dense_adamw_physical_gradient_paths",
]
