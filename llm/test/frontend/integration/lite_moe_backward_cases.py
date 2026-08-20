"""Self-contained production source case for S3-Lite MoE backward."""

from __future__ import annotations

from dataclasses import dataclass

from llm.test.frontend.integration.lite_moe_cases import (
    LiteMoeExecutionCase,
    build_lite_moe_execution_case,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_backward import (
    build_lite_moe_backward_overlay,
    validate_lite_moe_backward_overlay,
)
from llm.frontend.wafer_frontend.passes.lite_moe_backward_link_program import (
    link_lite_moe_backward_program,
)
from llm.frontend.wafer_frontend.passes.lite_moe_backward_lower_program import (
    lower_lite_moe_backward_program,
)
from llm.frontend.wafer_frontend.schema.lite_moe_backward import (
    S3_LITE_MOE_BACKWARD_CASE_ID,
    LiteMoeBackwardOverlay,
)
from llm.frontend.wafer_frontend.schema.lite_moe_backward_n6 import (
    LiteMoeBackwardLinkedProgram,
    LiteMoeBackwardLoweredProgram,
)


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardCase:
    """Exact integration wrapper; the production overlay is the sole truth."""

    forward: LiteMoeExecutionCase
    overlay: LiteMoeBackwardOverlay
    lowered: LiteMoeBackwardLoweredProgram
    linked: LiteMoeBackwardLinkedProgram

    @property
    def case_id(self) -> str:
        return self.overlay.contract.case_id

    def validate(self, path: str = "lite_moe_backward_case") -> None:
        self.forward.validate(f"{path}.forward")
        if self.case_id != S3_LITE_MOE_BACKWARD_CASE_ID:
            raise SchemaError("wrong MoE backward case id", path=f"{path}.case_id")
        validate_lite_moe_backward_overlay(
            self.overlay,
            self.forward.n4,
            self.forward.projection,
            self.forward.schedule,
            self.forward.global_dag,
            self.forward.n6_intent,
            self.forward.source.moe_spec.trace,
        )
        expected = build_lite_moe_backward_overlay(
            self.forward.n4,
            self.forward.projection,
            self.forward.schedule,
            self.forward.global_dag,
            self.forward.n6_intent,
            self.forward.source.moe_spec.trace,
        )
        if self.overlay != expected:
            raise SchemaError(
                "overlay must equal the exact production backward quotient",
                path=f"{path}.overlay",
            )
        expected_lowered = lower_lite_moe_backward_program(
            self.overlay,
            self.forward.n4,
            self.forward.projection,
            self.forward.schedule,
            self.forward.global_dag,
            self.forward.n6_intent,
            self.forward.source.moe_spec.trace,
        )
        if self.lowered != expected_lowered:
            raise SchemaError(
                "lowered carrier must equal the exact production backward quotient",
                path=f"{path}.lowered",
            )
        expected_linked = link_lite_moe_backward_program(self.lowered)
        if self.linked != expected_linked:
            raise SchemaError(
                "linked carrier must equal the exact production backward quotient",
                path=f"{path}.linked",
            )
        manifest = self.linked.manifest
        if (
            len(manifest.fragments),
            sum(
                len(stream.records)
                for fragment in manifest.fragments
                for stream in fragment.core_streams
            ),
            len(manifest.input_digests),
            len(manifest.core_streams),
            len(manifest.runtime_symbol_definitions),
            len(manifest.program_symbol_definitions),
            len(manifest.address_operand_bindings),
            len(manifest.state_operand_bindings),
        ) != (28, 104, 33, 2, 18, 69, 180, 8):
            raise SchemaError(
                "backward linked-manifest quotient changed",
                path=f"{path}.linked",
            )
        oracle = self.overlay.oracle
        if (
            len(self.overlay.remote_grad_dtes),
            len(self.overlay.token_wgrads),
            len(self.overlay.expert_reduces),
            len(self.overlay.sgd_stores),
            oracle.remote_grad_bytes_total,
            oracle.token_wgrad_bytes_total,
            oracle.sgd_store_count * oracle.state_store_bytes_each,
        ) != (4, 8, 4, 4, 128, 16384, 4096) or len(
            self.overlay.trainable_down_states
        ) != 4:
            raise SchemaError("backward production count/byte quotient changed", path=path)


def build_lite_moe_backward_case() -> LiteMoeBackwardCase:
    forward = build_lite_moe_execution_case()
    overlay = build_lite_moe_backward_overlay(
        forward.n4,
        forward.projection,
        forward.schedule,
        forward.global_dag,
        forward.n6_intent,
        forward.source.moe_spec.trace,
    )
    lowered = lower_lite_moe_backward_program(
        overlay,
        forward.n4,
        forward.projection,
        forward.schedule,
        forward.global_dag,
        forward.n6_intent,
        forward.source.moe_spec.trace,
    )
    linked = link_lite_moe_backward_program(lowered)
    result = LiteMoeBackwardCase(forward, overlay, lowered, linked)
    result.validate()
    return result


__all__ = [
    "LiteMoeBackwardCase",
    "build_lite_moe_backward_case",
]
