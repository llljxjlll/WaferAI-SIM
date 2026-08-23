from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_moe_scale import (
    MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION,
    MoeSwizzleExecutionStatus,
    MoeSwizzleScaleOracle,
    MoeSwizzleScaleSpec,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


class MoeSwizzleScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_moe_swizzle_scale_cases()

    def test_c0_c4_work_is_exact_and_production_derived(self) -> None:
        expected = (
            ("C0", 8, (2, 2, 2, 2), 6, 192, 24576, 256, 512, True),
            ("C1", 32, (8, 8, 8, 8), 24, 768, 98304, 1024, 2048, True),
            ("C2", 64, (16, 16, 16, 16), 48, 1536, 196608, 2048, 4096, True),
            ("C3", 128, (32, 32, 32, 32), 96, 3072, 393216, 4096, 8192, True),
            ("C4", 64, (32, 16, 8, 8), 48, 1536, 196608, 2048, 4096, False),
        )
        actual = tuple(
            (
                case.spec.name,
                case.oracle.assignment_count,
                case.oracle.expert_token_counts,
                len(case.oracle.remote_token_indices),
                case.oracle.dispatch_logical_bytes,
                case.oracle.total_expert_gemm_flops,
                case.oracle.combined_terminal_bytes,
                case.oracle.train_tape_terminal_bytes,
                case.spec.execution_status
                is MoeSwizzleExecutionStatus.PRODUCTION_READY,
            )
            for case in self.cases
        )
        self.assertEqual(actual, expected)
        c0 = self.cases[0]
        self.assertEqual(c0.spec.trace, c0.c0_production_case.spec.trace)
        self.assertEqual(c0.oracle.logical_p2p_bytes, 384)
        self.assertEqual(c0.oracle.data_packets, 24)

    def test_strict_serde_and_stable_ids(self) -> None:
        for case in self.cases:
            self.assertEqual(
                loads_dataclass(
                    MoeSwizzleScaleSpec,
                    canonical_json(case.spec),
                    path="spec",
                ),
                case.spec,
            )
            self.assertEqual(
                loads_dataclass(
                    MoeSwizzleScaleOracle,
                    canonical_json(case.oracle),
                    path="oracle",
                ),
                case.oracle,
            )

    def test_wrong_capacity_source_and_work_fail_closed(self) -> None:
        c0 = self.cases[0].c0_production_case
        assert c0 is not None
        c2 = self.cases[2]
        with self.assertRaisesRegex(SchemaError, "capacity"):
            replace(
                c2.spec,
                capacity_per_expert=c2.spec.capacity_per_expert + 1,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "token sources"):
            semantic = c2.spec._semantic_key()
            semantic["token_source_die_ids"] = (
                    (c2.spec.token_source_die_ids[0] + 1) % 4,
                    *c2.spec.token_source_die_ids[1:],
            )
            MoeSwizzleScaleSpec.create(**semantic).validate_against(
                c0.spec, c0.topology
            )
        with self.assertRaisesRegex(SchemaError, "exactly rebuild"):
            forged = replace(
                c2.oracle,
                dispatch_logical_bytes=c2.oracle.dispatch_logical_bytes + 32,
                logical_p2p_bytes=c2.oracle.logical_p2p_bytes + 32,
            )
            forged = replace(
                forged,
                id=stable_artifact_id(
                    "moe_swizzle_scale_oracle",
                    forged._semantic_key(),
                    schema_version=MOE_SWIZZLE_SCALE_ORACLE_SCHEMA_VERSION,
                ),
            )
            forged.validate_against(c2.spec)

    def test_capacity_point_cannot_claim_production_execution(self) -> None:
        c4 = self.cases[4]
        with self.assertRaisesRegex(SchemaError, "capacity probe"):
            replace(
                c4.spec,
                execution_status=MoeSwizzleExecutionStatus.PRODUCTION_READY,
            ).validate()


if __name__ == "__main__":
    unittest.main()
