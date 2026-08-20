"""Lower the exact S3-Lite MoE backward overlay into typed N6 leaves."""

from __future__ import annotations

from ..lowering.lite_moe_backward import lower_lite_moe_backward
from ..schema.lite_moe import LiteMoeStaticTrace
from ..schema.lite_moe_backward import LiteMoeBackwardOverlay
from ..schema.lite_moe_backward_n6 import LiteMoeBackwardLoweredProgram
from ..schema.lite_moe_execution import LiteMoeGlobalDag, LiteMoeProjection, LiteMoeScheduled
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.lite_moe_n6 import LiteMoeN6Intent


def lower_lite_moe_backward_program(
    overlay: LiteMoeBackwardOverlay,
    n4: LiteMoeN4IR1,
    projection: LiteMoeProjection,
    schedule: LiteMoeScheduled,
    global_dag: LiteMoeGlobalDag,
    n6_intent: LiteMoeN6Intent,
    trace: LiteMoeStaticTrace,
) -> LiteMoeBackwardLoweredProgram:
    result = LiteMoeBackwardLoweredProgram.create(
        n4=n4,
        projection=projection,
        schedule=schedule,
        global_dag=global_dag,
        n6_intent=n6_intent,
        trace=trace,
        overlay=overlay,
        fragments=lower_lite_moe_backward(
            overlay, n4, projection, schedule, global_dag, n6_intent, trace
        ),
    )
    result.validate()
    return result


__all__ = ["lower_lite_moe_backward_program"]
