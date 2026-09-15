"""Fail-closed admission for a real two-step full Dense AdamW training graph.

The current public AdamW runner is a WGRAD/optimizer leaf.  This gate makes
that limit executable: a later full runner must supply both source IR0/IR1
contexts and actual ordered forward/loss/all-layer-backward/AdamW records.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..errors import SchemaError
from ..lowering.context import LoweringContext
from ..schema.artifact_manifest import (
    LinkedProgramManifest, RecordOpcode, RegionManifest,
)
from ..schema.common import DType
from ..schema.ir0 import AdamwUpdateWorkload, IR0, JobKind, OpKind, OpPhase
from ..schema.persistent_state import StateKind


_REQUIRED = {
    RecordOpcode.EMBEDDING_LOOKUP: 1,
    RecordOpcode.RMSNORM: 1,
    RecordOpcode.CROSS_ENTROPY_FORWARD: 1,
    RecordOpcode.CROSS_ENTROPY_BACKWARD: 1,
    RecordOpcode.ADAMW_UPDATE: 17,
    RecordOpcode.LSU_LOAD: 83,
    RecordOpcode.LSU_STORE: 83,
}
_ROLES = {
    StateKind.TRAINABLE_PARAMETER: 15,
    StateKind.OPTIMIZER_MASTER: 17,
    StateKind.OPTIMIZER_MOMENT1: 17,
    StateKind.OPTIMIZER_MOMENT2: 17,
    StateKind.OPTIMIZER_STEP: 17,
}


@dataclass(frozen=True, slots=True)
class DenseAdamwFullTrainAdmission:
    source_ir0_ids: tuple[str, str]
    linked_manifest_ids: tuple[str, str]
    state_abi_ids: tuple[str, ...]
    adamw_updates_per_step: int
    physical_training_graph_complete: bool


def require_dense_adamw_full_train_admission(
    linked_steps: tuple[LinkedProgramManifest, LinkedProgramManifest],
    *,
    source_ir0_steps: tuple[IR0, IR0] | None = None,
    lowering_contexts: tuple[LoweringContext, LoweringContext] | None = None,
) -> DenseAdamwFullTrainAdmission:
    """Admit only production IR0→IR1/projection/schedule/linked full training.

    An opcode-count or optimizer-only StateABI inventory cannot override the
    deep validate_against source check.  Added CE/backward actions on a stale
    forward-only IR1 must therefore still fail.
    """
    if type(linked_steps) is not tuple or len(linked_steps) != 2 or any(
            type(item) is not LinkedProgramManifest for item in linked_steps):
        raise SchemaError("requires two independently linked AdamW steps",
                          path="dense_adamw_full_train.linked_steps")
    home_inventory = []
    for index, manifest in enumerate(linked_steps):
        manifest.validate(f"dense_adamw_full_train.step[{index}]")
        leaves = {
            (fragment.fragment.id if type(fragment) is RegionManifest else fragment.id):
            (fragment.fragment if type(fragment) is RegionManifest else fragment)
            for fragment in manifest.fragments
        }
        opcodes = []
        for stream in manifest.core_streams:
            for ref in stream.records:
                fragment = leaves[ref.fragment_id]
                local = next((part for part in fragment.core_streams
                              if part.logical_core == stream.logical_core), None)
                if local is None:
                    raise SchemaError("physical record lacks local core stream",
                                      path=f"dense_adamw_full_train.step[{index}]")
                opcodes.append(local.records[ref.fragment_record_index].opcode)
        counts = Counter(opcodes)
        missing = {opcode.name: count - counts[opcode]
                   for opcode, count in _REQUIRED.items()
                   if counts[opcode] < count}
        if not ({RecordOpcode.ATTENTION, RecordOpcode.ATTENTION_EXACT} & set(opcodes)):
            missing["ATTENTION"] = 1
        if missing:
            raise SchemaError(f"full forward/loss/backward/optimizer physical work missing: {missing}",
                              path=f"dense_adamw_full_train.step[{index}].records")
        ce_forward = opcodes.index(RecordOpcode.CROSS_ENTROPY_FORWARD)
        ce_backward = opcodes.index(RecordOpcode.CROSS_ENTROPY_BACKWARD)
        adamw_first = opcodes.index(RecordOpcode.ADAMW_UPDATE)
        if not ce_forward < ce_backward < adamw_first:
            raise SchemaError("AdamW ran before the complete forward/loss/backward edge",
                              path=f"dense_adamw_full_train.step[{index}].order")
        states = [state for fragment in leaves.values()
                  for state in fragment.state_abi if state.kind in _ROLES]
        kinds = Counter(state.kind for state in states)
        if any(kinds[kind] != count for kind, count in _ROLES.items()) or len(states) != 83:
            raise SchemaError("physical parameter/master/m/v/step StateABI inventory incomplete",
                              path=f"dense_adamw_full_train.step[{index}].state_abi")
        home_inventory.append(tuple(sorted(
            (state.state_ref, state.id, state.kind.value, state.die_id,
             state.address, state.size_bytes, state.dtype.value)
            for state in states)))
    if home_inventory[0] != home_inventory[1]:
        raise SchemaError("two steps do not share exact versioned AdamW StateABI homes",
                          path="dense_adamw_full_train.state_abi")
    if (source_ir0_steps is None or lowering_contexts is None or
            type(source_ir0_steps) is not tuple or len(source_ir0_steps) != 2 or
            type(lowering_contexts) is not tuple or len(lowering_contexts) != 2):
        raise SchemaError("full training needs both IR0 and deep IR1 lowering contexts",
                          path="dense_adamw_full_train.source")
    for index, (ir0, context, manifest) in enumerate(
            zip(source_ir0_steps, lowering_contexts, linked_steps)):
        if type(ir0) is not IR0 or type(context) is not LoweringContext:
            raise SchemaError("source must be typed IR0 and LoweringContext",
                              path=f"dense_adamw_full_train.source[{index}]")
        ir0.validate(f"dense_adamw_full_train.ir0[{index}]")
        context.validate(f"dense_adamw_full_train.context[{index}]")
        if ir0.job is not JobKind.TRAIN or context.ir1.source_ir0_id != ir0.id:
            raise SchemaError("training IR1 does not derive from supplied IR0",
                              path=f"dense_adamw_full_train.source[{index}]")
        kinds = Counter((node.kind, node.phase) for node in ir0.nodes)
        if (kinds[(OpKind.ATTENTION, OpPhase.FWD)] < 2 or
                kinds[(OpKind.CE_FORWARD, OpPhase.FWD)] != 1 or
                kinds[(OpKind.CE_BACKWARD, OpPhase.DGRAD)] != 1 or
                sum(count for (kind, phase), count in kinds.items()
                    if phase is OpPhase.WGRAD) < 17 or
                kinds[(OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE)] != 17):
            raise SchemaError("L2 full forward/loss/all-parameter backward/AdamW IR0 missing",
                              path=f"dense_adamw_full_train.ir0[{index}].nodes")
        updates = [node for node in ir0.nodes
                   if node.kind is OpKind.OPTIMIZER_UPDATE]
        if any(type(node.workload) is not AdamwUpdateWorkload or
               node.workload.step != index + 1
               for node in updates):
            raise SchemaError("optimizer IR0 versions are not 1 then 2",
                              path=f"dense_adamw_full_train.ir0[{index}].updates")
        values = {value.id: value for value in ir0.values}
        nodes = {node.id: node for node in ir0.nodes}
        gradient_refs = set()
        for node in updates:
            if len(node.inputs) != 6 or len(node.outputs) != 5:
                raise SchemaError("AdamW needs six physical reads/five writes",
                                  path=f"dense_adamw_full_train.ir0[{index}].updates")
            grad = values[node.inputs[1]]
            producer = nodes.get(grad.producer)
            if (grad.dtype is not DType.FP32 or producer is None or
                    producer.phase is not OpPhase.WGRAD or
                    producer.kind not in (
                        OpKind.GEMM_WEIGHT_WGRAD,
                        OpKind.NORM_GAMMA_WGRAD,
                        OpKind.EMBEDDING_TABLE_WGRAD) or
                    grad.id in gradient_refs):
                raise SchemaError("AdamW gradient is not a unique real WGRAD output",
                                  path=f"dense_adamw_full_train.ir0[{index}].updates")
            gradient_refs.add(grad.id)
        manifest.validate_against(
            context.ir1, context.fusion_plans, context.standalone_plans,
            context.projection, context.schedule_set, context.global_dag,
            manifest.fragments)
    return DenseAdamwFullTrainAdmission(
        tuple(ir0.id for ir0 in source_ir0_steps),
        tuple(manifest.id for manifest in linked_steps),
        tuple(item[1] for item in home_inventory[0]), 17, True)


__all__ = ["DenseAdamwFullTrainAdmission",
           "require_dense_adamw_full_train_admission"]
