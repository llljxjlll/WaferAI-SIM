"""Build and validate the fixed EP4 training-forward tape overlay."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir2 import BufferOwnership
from ..schema.lite_moe_dp4 import S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID
from ..schema.lite_moe_dp4_execution import LiteMoeDp4ExecutionCase
from ..schema.lite_moe_dp4_train_forward import (
    LiteMoeDp4TapeBuffer,
    LiteMoeDp4TapeCopy,
    LiteMoeDp4TrainForward,
)
from .lite_moe_dp4_execution import validate_lite_moe_dp4_execution_case


def _align64(value: int) -> int:
    return (value + 63) // 64 * 64


def _components(
    forward: LiteMoeDp4ExecutionCase,
) -> tuple[tuple[LiteMoeDp4TapeBuffer, ...], tuple[LiteMoeDp4TapeCopy, ...]]:
    validate_lite_moe_dp4_execution_case(forward)
    tasks = {
        task.id: task
        for projected in forward.projection.dies
        for task in projected.tasks
    }
    placements = {item.task_ref: item for item in forward.schedule.placements}
    actions = {item.id: item for item in forward.global_dag.actions}
    buffers = {
        (item.die_id, item.value_ref): item for item in forward.schedule.buffers
    }
    cursors = {
        die_id: _align64(
            max(
                item.address + item.size_bytes
                for item in forward.schedule.buffers
                if item.die_id == die_id
            )
        )
        for die_id in range(4)
    }
    ordinals = {
        projected.die_id: len(projected.tasks) for projected in forward.projection.dies
    }
    tape_buffers = []
    tape_copies = []
    assignments = forward.adapter.spec.trace.assignments
    for assignment in assignments:
        token = assignment.token_index
        expert = assignment.expert_index
        slot = token % 2
        prefix = f"S3M4.token{token}.expert{expert}"
        source_value_ref = f"{prefix}.value.swiglu"
        source_task_ref = f"moe.dp4.task.{prefix}.swiglu.comp"
        down_task_ref = f"moe.dp4.task.{prefix}.down.comp"
        source_action_ref = f"moe.dp4.action.{source_task_ref}"
        down_action_ref = f"moe.dp4.action.{down_task_ref}"
        source_task = tasks.get(source_task_ref)
        down_task = tasks.get(down_task_ref)
        source_action = actions.get(source_action_ref)
        down_action = actions.get(down_action_ref)
        source_buffer = buffers.get((expert, source_value_ref))
        placement = placements.get(source_task_ref)
        if (
            source_task is None
            or down_task is None
            or source_action is None
            or down_action is None
            or source_buffer is None
            or placement is None
            or source_task.write_values != (source_value_ref,)
            or source_value_ref not in down_task.read_values
            or source_task_ref not in down_task.deps
            or source_action_ref not in down_action.deps
            or placement.die_id != expert
        ):
            raise SchemaError(
                "training tape source is not a real SWIGLU-to-down fork",
                path=f"forward.token{token}",
            )
        tape = LiteMoeDp4TapeBuffer.create(
            token_index=token,
            expert_index=expert,
            slot_index=slot,
            die_id=expert,
            core_ref=placement.core_ref,
            value_ref=f"S3M4.train_forward.token{token}.expert{expert}.tape",
            address=cursors[expert],
            size_bytes=64,
            alignment_bytes=64,
            ownership=BufferOwnership.OWNED,
            terminal=True,
            alias_of=None,
            ordinal=ordinals[expert],
        )
        copy = LiteMoeDp4TapeCopy.create(
            token_index=token,
            expert_index=expert,
            slot_index=slot,
            die_id=expert,
            core_ref=placement.core_ref,
            source_value_ref=source_value_ref,
            source_task_ref=source_task_ref,
            source_action_ref=source_action_ref,
            source_buffer_ref=source_buffer.id,
            down_task_ref=down_task_ref,
            down_action_ref=down_action_ref,
            destination_buffer_ref=tape.id,
            bytes=64,
            dtype=DType.FP16,
            deps=(source_action_ref,),
        )
        tape_buffers.append(tape)
        tape_copies.append(copy)
        cursors[expert] += 64
        ordinals[expert] += 1
    return tuple(tape_buffers), tuple(tape_copies)


def build_lite_moe_dp4_train_forward(
    forward: LiteMoeDp4ExecutionCase,
) -> LiteMoeDp4TrainForward:
    tape_buffers, tape_copies = _components(forward)
    result = LiteMoeDp4TrainForward.create(
        case_id=S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
        source_topology_id=forward.adapter.topology.id,
        source_oracle_id=forward.adapter.oracle.id,
        forward=forward,
        tape_buffers=tape_buffers,
        tape_copies=tape_copies,
        total_tape_bytes=sum(item.size_bytes for item in tape_buffers),
    )
    validate_lite_moe_dp4_train_forward(result, forward)
    return result


def validate_lite_moe_dp4_train_forward(
    result: LiteMoeDp4TrainForward,
    forward: LiteMoeDp4ExecutionCase | None = None,
) -> None:
    result.validate()
    source = result.forward if forward is None else forward
    validate_lite_moe_dp4_execution_case(source)
    expected_buffers, expected_copies = _components(source)
    if (
        result.forward != source
        or result.source_topology_id != source.adapter.topology.id
        or result.source_oracle_id != source.adapter.oracle.id
        or result.tape_buffers != expected_buffers
        or result.tape_copies != expected_copies
        or result.total_tape_bytes != 512
    ):
        raise SchemaError(
            "training-forward is not the exact forward tape quotient",
            path="lite_moe_dp4_train_forward",
        )


__all__ = [
    "build_lite_moe_dp4_train_forward",
    "validate_lite_moe_dp4_train_forward",
]
