from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe import (
    build_lite_moe_oracle,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.lite_moe import (
    LITE_MOE_ORACLE_SCHEMA_VERSION,
    LITE_MOE_SPEC_SCHEMA_VERSION,
    LITE_MOE_TRACE_SCHEMA_VERSION,
    S3_LITE_BASELINE_EPOCH,
    S3_LITE_STATIC_MOE_CASE_ID,
    LiteMoeOracle,
    LiteMoeRoutingKind,
    LiteMoeSpec,
    LiteMoeStaticTrace,
    LiteMoeTraceAssignment,
    LiteMoeTransferRole,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)


def _trace() -> LiteMoeStaticTrace:
    assignments = tuple(
        LiteMoeTraceAssignment(
            token_index=token_index,
            expert_index=(token_index // 2) % 4,
            slot_index=token_index % 2,
        )
        for token_index in range(8)
    )
    return LiteMoeStaticTrace.create(
        token_count=8,
        assignments=assignments,
        expert_histogram=(2, 2, 2, 2),
    )


def _spec() -> LiteMoeSpec:
    return LiteMoeSpec.create(
        hidden_size=16,
        intermediate_size=32,
        capacity_per_expert=2,
        trace=_trace(),
    )


class LiteMoeTest(unittest.TestCase):
    def test_balanced_tiny_golden_round_trip_and_determinism(self) -> None:
        spec = _spec()
        oracle = build_lite_moe_oracle(spec)
        self.assertEqual(S3_LITE_BASELINE_EPOCH, "s3-lite-v1")
        self.assertEqual(
            S3_LITE_STATIC_MOE_CASE_ID,
            "case.s3_lite.static_moe_infer",
        )
        self.assertEqual(
            LITE_MOE_TRACE_SCHEMA_VERSION,
            "wafer_frontend.s3_lite_static_moe_trace/v1alpha1",
        )
        self.assertEqual(
            LITE_MOE_SPEC_SCHEMA_VERSION,
            "wafer_frontend.s3_lite_static_moe_spec/v1alpha1",
        )
        self.assertEqual(
            LITE_MOE_ORACLE_SCHEMA_VERSION,
            "wafer_frontend.s3_lite_static_moe_oracle/v1alpha1",
        )
        self.assertEqual(spec.trace.expert_histogram, (2, 2, 2, 2))
        self.assertEqual(
            tuple(metric.token_count for metric in oracle.expert_metrics),
            (2, 2, 2, 2),
        )
        self.assertEqual(
            tuple(metric.gemm_flops for metric in oracle.expert_metrics),
            (6144, 6144, 6144, 6144),
        )
        self.assertEqual(oracle.total_expert_gemm_flops, 24576)
        self.assertEqual(oracle.logical_p2p_bytes, 256)
        self.assertEqual(oracle.per_hop_p2p_bytes, 256)
        self.assertEqual(
            tuple(
                (
                    metric.role,
                    metric.source_die_id,
                    metric.destination_die_id,
                    metric.token_count,
                    metric.logical_bytes,
                )
                for metric in oracle.p2p_metrics
            ),
            (
                (LiteMoeTransferRole.MOE_DISPATCH, 0, 1, 2, 64),
                (LiteMoeTransferRole.MOE_DISPATCH, 1, 0, 2, 64),
                (LiteMoeTransferRole.MOE_COMBINE, 0, 1, 2, 64),
                (LiteMoeTransferRole.MOE_COMBINE, 1, 0, 2, 64),
            ),
        )
        oracle.validate_against(spec)
        self.assertEqual(build_lite_moe_oracle(spec), oracle)
        for value, value_type in (
            (spec.trace, LiteMoeStaticTrace),
            (spec, LiteMoeSpec),
            (oracle, LiteMoeOracle),
        ):
            self.assertEqual(
                loads_dataclass(
                    value_type,
                    canonical_json(value),
                    path="round_trip",
                ),
                value,
            )

    def test_trace_coverage_slots_histogram_and_serde_fail_closed(self) -> None:
        trace = _trace()
        duplicate_token = replace(
            trace.assignments[1], token_index=0
        )
        with self.assertRaisesRegex(SchemaError, "cover every token"):
            replace(
                trace,
                assignments=(trace.assignments[0], duplicate_token, *trace.assignments[2:]),
            ).validate()
        slot_gap = replace(trace.assignments[1], slot_index=2)
        with self.assertRaisesRegex(SchemaError, "gap-free"):
            replace(
                trace,
                assignments=(trace.assignments[0], slot_gap, *trace.assignments[2:]),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "assignment histogram"):
            replace(trace, expert_histogram=(1, 3, 2, 2)).validate()
        with self.assertRaisesRegex(SchemaError, "one assignment per token"):
            replace(trace, assignments=trace.assignments[:-1]).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(trace, schema_version="v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(trace, id="forged").validate()

        raw = json.loads(canonical_json(trace))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(LiteMoeStaticTrace, json.dumps(raw), path="trace")
        del raw["unexpected"]
        del raw["assignments"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(LiteMoeStaticTrace, json.dumps(raw), path="trace")

    def test_fixed_scope_random_gate_topk_overflow_and_drop_rejected(self) -> None:
        spec = _spec()
        for field_name, value, pattern in (
            ("die_count", 3, "exactly 2"),
            ("ep_degree", 1, "exactly 2"),
            ("expert_count", 8, "exactly 4"),
            ("top_k", 2, "exactly 1"),
            ("routing_kind", LiteMoeRoutingKind.RANDOM, "only STATIC_TRACE"),
            ("routing_kind", LiteMoeRoutingKind.GATE_TOPK, "GATE_TOPK"),
            ("allow_overflow", True, "forbids overflow"),
            ("drop_tokens", True, "token dropping"),
            ("dtype", DType.FP32, "only FP16"),
            ("capacity_per_expert", 1, "exceeds expert capacity"),
        ):
            with self.subTest(field=field_name, value=value):
                with self.assertRaisesRegex(SchemaError, pattern):
                    replace(spec, **{field_name: value}).validate()
        with self.assertRaisesRegex(SchemaError, "case.s3_lite"):
            replace(spec, case_id="case.other").validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(spec, schema_version="v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(spec, id="forged").validate()
        raw = json.loads(canonical_json(spec))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(LiteMoeSpec, json.dumps(raw), path="spec")
        del raw["unexpected"]
        del raw["trace"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(LiteMoeSpec, json.dumps(raw), path="spec")

    def test_oracle_formula_route_and_source_tamper_fail_closed(self) -> None:
        spec = _spec()
        oracle = build_lite_moe_oracle(spec)
        changed_expert = replace(
            oracle.expert_metrics[0], gemm_flops=6145
        )
        bad_work = LiteMoeOracle.create(
            **(
                oracle._semantic_key()
                | {
                    "expert_metrics": (
                        changed_expert,
                        *oracle.expert_metrics[1:],
                    ),
                    "total_expert_gemm_flops": 24577,
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "independently recomputed"):
            bad_work.validate_against(spec)

        changed_route = replace(
            oracle.p2p_metrics[0],
            logical_bytes=80,
            byte_hop_bytes=80,
        )
        bad_route = LiteMoeOracle.create(
            **(
                oracle._semantic_key()
                | {
                    "p2p_metrics": (changed_route, *oracle.p2p_metrics[1:]),
                    "logical_p2p_bytes": 272,
                    "per_hop_p2p_bytes": 272,
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "independently recomputed"):
            bad_route.validate_against(spec)
        with self.assertRaisesRegex(SchemaError, "one hop"):
            replace(
                oracle.p2p_metrics[0], hop_count=2
            ).validate()

        foreign = LiteMoeOracle.create(
            **(oracle._semantic_key() | {"source_spec_digest": "2" * 64})
        )
        with self.assertRaisesRegex(SchemaError, "supplied spec"):
            foreign.validate_against(spec)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(oracle, schema_version="v0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(oracle, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "role/endpoint order"):
            replace(
                oracle, p2p_metrics=tuple(reversed(oracle.p2p_metrics))
            ).validate()
        raw = json.loads(canonical_json(oracle))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(LiteMoeOracle, json.dumps(raw), path="oracle")
        del raw["unexpected"]
        del raw["p2p_metrics"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(LiteMoeOracle, json.dumps(raw), path="oracle")


if __name__ == "__main__":
    unittest.main()
