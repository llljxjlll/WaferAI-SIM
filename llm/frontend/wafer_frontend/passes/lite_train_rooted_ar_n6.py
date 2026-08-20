"""Producer for the exact S2-Lite DP2 rooted-AR N6 intent."""

from __future__ import annotations

from ..errors import SchemaError
from ..lowering.coarse import _buffer_abi
from ..schema.artifact_manifest import COMMAND_FRAGMENT_SCHEMA_VERSION, BufferABI
from ..schema.common import DType, stable_artifact_id
from ..schema.ir2 import BufferOwnership, BufferUseRole, TensorSlice
from ..schema.lite_train_dp2 import RootedArStepKind, S2LiteDp2RootedArGlobalAction
from ..schema.lite_train_rooted_ar_n6 import (
    RootedArExecutableKind,
    RootedArExecutableUnit,
    S2LiteRootedArN6Intent,
    rooted_ar_lowering_contexts,
)


def _gradient_abi(context, marker: str) -> BufferABI:
    action = next(action for action in context.global_dag.actions if marker in getattr(action.source, "task_id", ""))
    use = next(use for use in action.buffer_uses if use.role is BufferUseRole.COMP_OUTPUT)
    schedule = next(schedule for schedule in context.schedule_set.schedules if schedule.die_id == action.logical_core.die_id)
    binding = next(binding for binding in schedule.buffer_bindings if binding.id == use.binding_id)
    return _buffer_abi(schedule.id, binding, action.logical_core)


def _scratch_abi(source, context, source_slice, offset: int, lifetime_end: int) -> BufferABI:
    gradient = _gradient_abi(context, ".lm_head_wgrad")
    schedule = next(schedule for schedule in context.schedule_set.schedules if schedule.die_id == 0)
    value_id = f"s2_lite.rooted_ar.scratch.{source_slice.replica_index}"
    tensor_slice = TensorSlice(value_id, (0,), (512,))
    semantic = {
        "schedule_id": schedule.id,
        "binding_id": source_slice.id,
        "value_id": value_id,
        "logical_core": gradient.logical_core,
        "tensor_slice": tensor_slice,
        "region_ref": gradient.region_ref,
        "region_offset_bytes": offset,
        "size_bytes": source_slice.size_bytes,
        "alignment_bytes": source.ar_contract.alignment_bytes,
        "banks": (),
        "storage_id": f"s2_lite.rooted_ar.storage.{source_slice.id}",
        "alias_of": None,
        "lifetime_start": gradient.lifetime_start,
        "lifetime_end_exclusive": lifetime_end,
        "dtype": DType.FP32,
        "layout": "row_major_rooted_ar_scratch",
        "ownership": BufferOwnership.OWNED,
    }
    return BufferABI(
        stable_artifact_id("buffer_abi", semantic, schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION),
        **semantic,
    )


def build_s2_lite_rooted_ar_n6_intent(
    source: S2LiteDp2RootedArGlobalAction,
) -> S2LiteRootedArN6Intent:
    if type(source) is not S2LiteDp2RootedArGlobalAction:
        raise SchemaError("must be an S2LiteDp2RootedArGlobalAction", path="source")
    source.validate("source")
    contexts = rooted_ar_lowering_contexts(source)
    gradients = tuple(_gradient_abi(context, ".lm_head_wgrad") for context in contexts)
    root_schedule = next(schedule for schedule in contexts[0].schedule_set.schedules if schedule.die_id == 0)
    scratch_base = 32768
    scratches = tuple(
        _scratch_abi(source, contexts[0], source_slice, scratch_base + source_slice.offset_bytes, len(root_schedule.placements) + 8)
        for source_slice in source.ar_contract.root_input_slices
    )
    upload, root_reduce, download = source.ar_steps
    root_core, leaf_core = gradients[0].logical_core, gradients[1].logical_core
    unit_specs = (
        (RootedArExecutableKind.LOCAL_COPY, root_reduce.id, root_core, (gradients[0].id,), scratches[0].id, (root_reduce.deps[0],)),
        (RootedArExecutableKind.UPLOAD_SEND, upload.id, leaf_core, (gradients[1].id,), None, upload.deps),
        (RootedArExecutableKind.UPLOAD_RECV, upload.id, root_core, (), scratches[1].id, ()),
    )
    units = [RootedArExecutableUnit.create(kind=k, source_step_ref=s, logical_core=c, input_buffer_abi_refs=i, output_buffer_abi_ref=o, bytes=2048, deps=d) for k,s,c,i,o,d in unit_specs]
    units.append(RootedArExecutableUnit.create(kind=RootedArExecutableKind.UPLOAD_WAIT, source_step_ref=upload.id, logical_core=root_core, input_buffer_abi_refs=(), output_buffer_abi_ref=None, bytes=2048, deps=(units[2].id,)))
    units.append(RootedArExecutableUnit.create(kind=RootedArExecutableKind.ROOT_REDUCE, source_step_ref=root_reduce.id, logical_core=root_core, input_buffer_abi_refs=(scratches[0].id, scratches[1].id), output_buffer_abi_ref=scratches[0].id, bytes=2048, deps=(units[0].id, units[3].id)))
    units.append(RootedArExecutableUnit.create(kind=RootedArExecutableKind.DOWNLOAD_SEND, source_step_ref=download.id, logical_core=root_core, input_buffer_abi_refs=(scratches[0].id,), output_buffer_abi_ref=None, bytes=2048, deps=(units[4].id,)))
    units.append(RootedArExecutableUnit.create(kind=RootedArExecutableKind.DOWNLOAD_RECV, source_step_ref=download.id, logical_core=leaf_core, input_buffer_abi_refs=(), output_buffer_abi_ref=gradients[1].id, bytes=2048, deps=()))
    units.append(RootedArExecutableUnit.create(kind=RootedArExecutableKind.DOWNLOAD_WAIT, source_step_ref=download.id, logical_core=leaf_core, input_buffer_abi_refs=(), output_buffer_abi_ref=None, bytes=2048, deps=(units[6].id,)))
    return S2LiteRootedArN6Intent.create(
        source=source,
        lowering_contexts=contexts,
        gradient_buffer_abis=gradients,
        scratch_buffer_abis=scratches,
        units=tuple(units),
    )


__all__ = ["build_s2_lite_rooted_ar_n6_intent"]
