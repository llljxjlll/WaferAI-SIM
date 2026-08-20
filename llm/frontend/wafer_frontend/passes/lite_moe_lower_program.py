"""Production fragment lowering for the S3-Lite MoE intent."""

from __future__ import annotations

from ..lowering.lite_moe import lower_lite_moe_n6_intent
from ..schema.lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjection,
    LiteMoeScheduled,
)
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.lite_moe_n6 import LiteMoeLoweredProgram, LiteMoeN6Intent
from .lite_moe_n6 import validate_lite_moe_n6_intent


def lower_lite_moe_n6(
    intent: LiteMoeN6Intent,
    global_dag: LiteMoeGlobalDag,
    schedule: LiteMoeScheduled,
    projection: LiteMoeProjection,
    n4: LiteMoeN4IR1,
) -> LiteMoeLoweredProgram:
    validate_lite_moe_n6_intent(intent, global_dag, schedule, projection, n4)
    return LiteMoeLoweredProgram.create(
        n4=n4,
        projection=projection,
        schedule=schedule,
        global_dag=global_dag,
        intent=intent,
        fragments=lower_lite_moe_n6_intent(intent, global_dag, schedule, n4),
    )


__all__ = ["lower_lite_moe_n6"]
