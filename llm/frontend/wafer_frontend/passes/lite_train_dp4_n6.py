"""Producer for the fixed S2-Lite DP4 tree-AllReduce N6 intent."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.coarse import _buffer_abi
from ..schema.artifact_manifest import COMMAND_FRAGMENT_SCHEMA_VERSION, BufferABI
from ..schema.common import DType, stable_artifact_id
from ..schema.ir2 import BufferOwnership, BufferUseRole, TensorSlice
from ..schema.lite_train_dp4 import S2LiteDp4TreeArGlobalAction
from ..schema.lite_train_dp4_n6 import (
    S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
    S2LiteDp4TreeArN6Intent,
    dp4_tree_lowering_contexts,
    dp4_tree_units,
)


def _gradient_abi(context) -> BufferABI:
    action = next(
        action
        for action in context.global_dag.actions
        if ".lm_head_wgrad" in getattr(action.source, "task_id", "")
    )
    use = next(use for use in action.buffer_uses if use.role is BufferUseRole.COMP_OUTPUT)
    schedule = next(
        schedule
        for schedule in context.schedule_set.schedules
        if schedule.die_id == action.logical_core.die_id
    )
    binding = next(binding for binding in schedule.buffer_bindings if binding.id == use.binding_id)
    return _buffer_abi(schedule.id, binding, action.logical_core)


def _scratch_abi(
    source: S2LiteDp4TreeArGlobalAction,
    context,
    gradient: BufferABI,
    *,
    die_id: int,
    slice_index: int,
    region_offset_bytes: int,
) -> BufferABI:
    schedule = next(
        item for item in context.schedule_set.schedules if item.die_id == die_id
    )
    binding_semantic = {
        "case_id": source.tree_contract.case_id,
        "die_id": die_id,
        "slice_index": slice_index,
        "offset_bytes": slice_index * 2048,
        "size_bytes": 2048,
    }
    binding_id = stable_artifact_id(
        "s2_lite_dp4_tree_scratch_slice",
        binding_semantic,
        schema_version=S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
    )
    value_id = f"s2_lite.dp4.tree.scratch.die{die_id}.slice{slice_index}"
    tensor_slice = TensorSlice(value_id, (0,), (512,))
    semantic = {
        "schedule_id": schedule.id,
        "binding_id": binding_id,
        "value_id": value_id,
        "logical_core": gradient.logical_core,
        "tensor_slice": tensor_slice,
        "region_ref": gradient.region_ref,
        "region_offset_bytes": region_offset_bytes,
        "size_bytes": 2048,
        "alignment_bytes": source.tree_contract.alignment_bytes,
        "banks": (),
        "storage_id": f"s2_lite.dp4.tree.storage.die{die_id}.slice{slice_index}",
        "alias_of": None,
        "lifetime_start": gradient.lifetime_start,
        "lifetime_end_exclusive": len(schedule.placements) + 23,
        "dtype": DType.FP32,
        "layout": "row_major_dp4_tree_scratch",
        "ownership": BufferOwnership.OWNED,
    }
    return BufferABI(
        stable_artifact_id(
            "buffer_abi", semantic, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION
        ),
        **semantic,
    )


def build_s2_lite_dp4_tree_ar_n6_intent(
    source: S2LiteDp4TreeArGlobalAction,
) -> S2LiteDp4TreeArN6Intent:
    if type(source) is not S2LiteDp4TreeArGlobalAction:
        raise SchemaError("must be an S2LiteDp4TreeArGlobalAction", path="source")
    source.validate("source")
    contexts = dp4_tree_lowering_contexts(source)
    gradients = tuple(_gradient_abi(context) for context in contexts)
    scratches = tuple(
        _scratch_abi(
            source,
            contexts[die_id],
            gradients[die_id],
            die_id=die_id,
            slice_index=slice_index,
            region_offset_bytes=32768 + slice_index * 2048,
        )
        for die_id in (0, 2)
        for slice_index in (0, 1)
    )
    return S2LiteDp4TreeArN6Intent.create(
        source=source,
        lowering_contexts=contexts,
        gradient_buffer_abis=gradients,
        scratch_buffer_abis=scratches,
        units=dp4_tree_units(source, gradients, scratches),
    )


__all__ = ["build_s2_lite_dp4_tree_ar_n6_intent"]
