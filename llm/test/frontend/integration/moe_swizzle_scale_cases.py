"""Production-derived C0-C4 workload truth for MoE Swizzle V2."""

from __future__ import annotations

from dataclasses import dataclass
import json

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_scale import (
    build_moe_swizzle_scale_truth,
    validate_moe_swizzle_c0_oracle,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeEndpointSessionContract,
    MoeHardwareFacts,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_scale import (
    MoeSwizzleExecutionStatus,
    MoeSwizzleScaleOracle,
    MoeSwizzleScaleSpec,
)

from lite_moe_dp4_cases import (
    LiteMoeDp4Case,
    LiteMoeDp4Mode,
    build_lite_moe_dp4_case,
)


@dataclass(frozen=True, slots=True)
class MoeSwizzleScaleCase:
    spec: MoeSwizzleScaleSpec
    oracle: MoeSwizzleScaleOracle
    c0_production_case: LiteMoeDp4Case | None
    hardware_facts: MoeHardwareFacts
    endpoint_session_contract: MoeEndpointSessionContract

    def validate(
        self,
        c0: LiteMoeDp4Case,
        path: str = "moe_swizzle_scale_case",
    ) -> None:
        c0.validate(f"{path}.c0")
        if c0.mode is not LiteMoeDp4Mode.INFER:
            raise SchemaError("C0 source must be inference production truth", path=f"{path}.c0")
        self.spec.validate_against(c0.spec, c0.topology, f"{path}.spec")
        self.oracle.validate_against(self.spec, f"{path}.oracle")
        self.hardware_facts.validate(f"{path}.hardware_facts")
        self.endpoint_session_contract.validate(f"{path}.endpoint_session_contract")
        if self.spec.name == "C0":
            if self.c0_production_case != c0:
                raise SchemaError("C0 scale must bind exact legacy production", path=path)
            validate_moe_swizzle_c0_oracle(
                self.oracle,
                c0.oracle,
                c0.spec,
                c0.topology,
            )
        elif self.c0_production_case is not None:
            raise SchemaError("only C0 binds the legacy production carrier", path=path)


def build_moe_swizzle_scale_cases() -> tuple[MoeSwizzleScaleCase, ...]:
    """Build C0-C4 from one canonical production LiteMoE DP4 carrier."""

    c0 = build_lite_moe_dp4_case(LiteMoeDp4Mode.INFER)
    truth = build_moe_swizzle_scale_truth(c0.spec, c0.topology)
    hardware_facts = MoeHardwareFacts.from_fabric(
        physical_fabric_from_data(json.loads(c0.hardware_json), path="moe_swizzle_scale_case.hardware")
    )
    endpoint_session_contract = MoeEndpointSessionContract.production()
    result = []
    for spec, oracle in truth:
        case = MoeSwizzleScaleCase(
            spec,
            oracle,
            c0 if spec.name == "C0" else None,
            hardware_facts,
            endpoint_session_contract,
        )
        case.validate(c0)
        result.append(case)
    return tuple(result)


__all__ = ["MoeSwizzleScaleCase", "build_moe_swizzle_scale_cases"]
