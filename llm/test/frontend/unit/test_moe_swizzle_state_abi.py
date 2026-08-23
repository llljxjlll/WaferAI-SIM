from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
    validate_moe_swizzle_workload_state_abi_against,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_state import (
    MoeSwizzleWorkloadStateABI,
    parse_moe_expert_weight_tensor_ref,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


class MoeSwizzleStateAbiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_moe_swizzle_scale_cases()[0]
        cls.execution = build_moe_swizzle_execution(
            cls.case.spec,
            cls.case.oracle,
            MoeScaleExecutionMode.INFER_FORWARD,
        )
        cls.ir1 = cls.case.c0_production_case.forward.n4.graph
        cls.abi = build_moe_swizzle_workload_state_abi(
            cls.ir1, cls.execution, cls.case.spec, cls.case.oracle,
        )

    def test_exact_state_dma_closure_and_determinism(self) -> None:
        self.assertEqual(len(self.abi.tensors), 12)
        self.assertEqual(len(self.abi.action_bindings), 3 * self.case.spec.tokens)
        self.assertEqual(
            {parse_moe_expert_weight_tensor_ref(item.tensor_ref) for item in self.abi.tensors},
            {(expert, role) for expert in range(4) for role in ("gate", "up", "down")},
        )
        self.assertEqual(
            build_moe_swizzle_workload_state_abi(
                self.ir1, self.execution, self.case.spec, self.case.oracle,
            ),
            self.abi,
        )
        validate_moe_swizzle_workload_state_abi_against(
            self.abi, self.ir1, self.execution, self.case.spec, self.case.oracle,
        )
        first_binding = self.abi.action_bindings[0]
        forged_binding = replace(first_binding, staging_value_ref="forged.staging")
        forged = type(self.abi).create(
            source_ir1_id=self.abi.source_ir1_id,
            source_manifest_id=self.abi.source_manifest_id,
            source_execution_id=self.abi.source_execution_id,
            hidden_size=self.abi.hidden_size,
            intermediate_size=self.abi.intermediate_size,
            tensors=self.abi.tensors,
            action_bindings=(forged_binding, *self.abi.action_bindings[1:]),
        )
        with self.assertRaisesRegex(SchemaError, "deterministic persistent-state rebuild"):
            validate_moe_swizzle_workload_state_abi_against(
                forged, self.ir1, self.execution, self.case.spec, self.case.oracle,
            )
        self.assertEqual(
            loads_dataclass(
                MoeSwizzleWorkloadStateABI,
                canonical_json(self.abi),
                path="state_abi",
            ),
            self.abi,
        )

    def test_tensor_identity_missing_and_duplicate_fail_closed(self) -> None:
        for value in (
            "S3M4.expert0.gate",
            "S3M4.expert4.weight.gate",
            "S3M4.expert0.weight.bias",
            "S3M4.expert0.weight.gate.extra",
        ):
            with self.assertRaises(SchemaError):
                parse_moe_expert_weight_tensor_ref(value)
        with self.assertRaises(SchemaError):
            replace(self.abi, tensors=self.abi.tensors[:-1]).validate()
        with self.assertRaises(SchemaError):
            replace(
                self.abi,
                tensors=(*self.abi.tensors[:-1], self.abi.tensors[0]),
            ).validate()

    def test_home_and_address_tamper_fail_closed(self) -> None:
        first = self.abi.tensors[0]
        with self.assertRaises(SchemaError):
            replace(first, home_die_id=(first.home_die_id + 1) % 4).validate("tensor")
        with self.assertRaises(SchemaError):
            replace(first, hbm_address=first.hbm_address + 1).validate("tensor")
        with self.assertRaises(SchemaError):
            replace(
                first,
                hbm_address=first.address_space_base + first.address_space_size,
            ).validate("tensor")


if __name__ == "__main__":
    unittest.main()
