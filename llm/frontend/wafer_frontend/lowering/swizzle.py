"""Exact opcode lowering for the independent production Swizzle projection."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.swizzle_ir2 import SwizzleIr2Projection
from ..schema.swizzle_lowering import (
    SwizzleLoweredProgram,
    SwizzleLoweredRankStream,
    SwizzleOpcodeRecord,
    expected_swizzle_opcodes,
    validate_swizzle_plan_projection,
)
from ..schema.swizzle_plan import SwizzleFusionPlan


def lower_swizzle_projection(
    plan: SwizzleFusionPlan,
    projection: SwizzleIr2Projection,
) -> SwizzleLoweredProgram:
    """Lower task kinds to existing opcodes without inventing ABI operands."""

    validate_swizzle_plan_projection(plan, projection)
    actions = {
        action.source_action.id: action
        for program in plan.rank_programs
        for action in program.actions
    }
    streams = tuple(
        SwizzleLoweredRankStream(
            rank=dag.rank,
            die_id=dag.die_id,
            records=tuple(
                SwizzleOpcodeRecord.create(
                    task=task,
                    opcodes=expected_swizzle_opcodes(
                        actions[task.source_action_ref],
                        task,
                    ),
                )
                for task in dag.tasks
            ),
        )
        for dag in projection.rank_dags
    )
    if not streams:
        raise SchemaError("projection has no rank streams", path="projection.rank_dags")
    result = SwizzleLoweredProgram.create(
        producer_pass="swizzle_opcode_lowering",
        source_plan_ref=plan.id,
        source_projection_ref=projection.id,
        source_ir1_id=plan.source_ir1_id,
        source_decision_ref=plan.decision.id,
        source_candidate_ref=plan.candidate.id,
        pattern=plan.pattern,
        algorithm=plan.algorithm,
        rank_streams=streams,
        timing_execution=True,
        functional_execution=False,
    )
    result.validate_against(plan, projection)
    return result


__all__ = ["lower_swizzle_projection"]
