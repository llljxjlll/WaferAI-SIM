from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes.lite_moe import (
    build_lite_moe_oracle,
)
from llm.frontend.wafer_frontend.schema.lite_moe import LiteMoeSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from llm.test.frontend.integration.lite_moe_cases import (
    build_lite_moe_execution_case,
    build_lite_moe_source_case,
)


class LiteMoeSourceCaseTest(unittest.TestCase):
    def test_execution_case_is_exact_deterministic_and_provenanced(self) -> None:
        first = build_lite_moe_execution_case()
        second = build_lite_moe_execution_case()
        first.validate()
        self.assertEqual(
            canonical_digest(first), canonical_digest(second)
        )
        self.assertEqual(len(first.adapter.graph.nodes), 40)
        self.assertEqual(len(first.global_dag.actions), 80)
        self.assertEqual(len(first.n6_intent.buffer_abis), 64)
        self.assertEqual(len(first.n6_intent.state_loads), 24)
        self.assertEqual(len(first.n6_intent.compute_units), 32)
        self.assertEqual(len(first.n6_intent.dte_units), 8)
        self.assertEqual(first.placed.source_adapter_id, first.adapter.id)
        self.assertEqual(first.n4.source_placed_id, first.placed.id)
        self.assertEqual(first.projection.source_n4_id, first.n4.id)
        self.assertEqual(
            first.schedule.source_projection_id, first.projection.id
        )
        self.assertEqual(
            first.global_dag.source_schedule_id, first.schedule.id
        )
        self.assertEqual(
            first.n6_intent.source_global_id, first.global_dag.id
        )
        forged_intent = first.n6_intent.create(
            **(
                first.n6_intent._semantic_key()
                | {"source_global_id": "forged"}
            )
        )
        with self.assertRaisesRegex(SchemaError, "lowering quotient"):
            replace(
                first,
                n6_intent=forged_intent,
            ).validate()

    def test_source_case_is_exact_and_deterministic(self) -> None:
        first = build_lite_moe_source_case()
        second = build_lite_moe_source_case()
        first.validate()

        self.assertEqual(first, second)
        self.assertEqual(first.spec.model.V, 32)
        self.assertEqual(first.spec.model.H, 16)
        self.assertEqual(first.spec.model.I, 32)
        self.assertEqual(first.spec.model.L, 1)
        self.assertEqual(first.moe_spec.ep_degree, 2)
        self.assertEqual(first.moe_spec.expert_count, 4)
        self.assertEqual(first.moe_spec.top_k, 1)
        self.assertEqual(first.moe_spec.trace.expert_histogram, (2, 2, 2, 2))
        self.assertEqual(first.oracle.total_expert_gemm_flops, 24576)
        self.assertEqual(first.oracle.logical_p2p_bytes, 256)
        self.assertEqual(first.oracle.per_hop_p2p_bytes, 256)
        self.assertEqual(
            sum(
                metric.token_count
                for metric in first.oracle.p2p_metrics
                if metric.role.value == "moe_dispatch"
            ),
            4,
        )
        self.assertEqual(first.mapping_text, "0:0\n")
        self.assertEqual(
            canonical_digest(first.spec), canonical_digest(second.spec)
        )
        self.assertEqual(
            canonical_digest(first.moe_spec), canonical_digest(second.moe_spec)
        )
        self.assertEqual(
            canonical_digest(first.oracle), canonical_digest(second.oracle)
        )

    def test_outer_shape_oracle_and_case_identity_tamper_fail_closed(self) -> None:
        case = build_lite_moe_source_case()
        wrong_shape = LiteMoeSpec.create(
            hidden_size=8,
            intermediate_size=case.moe_spec.intermediate_size,
            capacity_per_expert=case.moe_spec.capacity_per_expert,
            trace=case.moe_spec.trace,
        )
        with self.assertRaisesRegex(SchemaError, "H/I"):
            replace(
                case,
                moe_spec=wrong_shape,
                oracle=build_lite_moe_oracle(wrong_shape),
            ).validate()
        foreign = case.oracle.create(
            **(
                case.oracle._semantic_key()
                | {"source_spec_digest": "1" * 64}
            )
        )
        with self.assertRaisesRegex(SchemaError, "supplied spec"):
            replace(case, oracle=foreign).validate()
        with self.assertRaisesRegex(SchemaError, "case.s3_lite"):
            replace(
                case,
                moe_spec=replace(case.moe_spec, case_id="case.other"),
            ).validate()

    def test_hardware_and_mapping_must_remain_two_die_and_usable(self) -> None:
        case = build_lite_moe_source_case()
        one_die = json.loads(case.hardware_json)
        one_die["die"] = {"x": 1, "y": 1}
        with self.assertRaises(SchemaError):
            replace(
                case,
                hardware_json=json.dumps(one_die),
            ).validate()
        with self.assertRaisesRegex(UnsupportedFeatureError, "identity mapping"):
            replace(case, mapping_text="0:1\n").validate()
        with self.assertRaisesRegex(SchemaError, "hardware JSON text"):
            replace(case, hardware_json=object()).validate()


if __name__ == "__main__":
    unittest.main()
