from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes import (
    build_stage3_dense_inference_oracle,
    logical_expand,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec, InferOutput
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    LogicalRole,
    OpKind,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.logical import IR0Template
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage3_dense_inference_oracle import (
    Stage3DenseInferenceOracle,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3ProfileMode,
    Stage3StaticProfile,
    StaticRequestShape,
)

from _fixtures import valid_spec


def _requests(
    entries: tuple[tuple[str, int, int, int], ...],
) -> tuple[StaticRequestShape, ...]:
    result: list[StaticRequestShape] = []
    page_start = 0
    for request_ref, prefill, decode, context in entries:
        page_count = (context + 15) // 16
        result.append(
            StaticRequestShape(
                request_ref=request_ref,
                prefill_tokens=prefill,
                decode_tokens=decode,
                context_tokens=context,
                kv_span=KvPageSpan(page_start, page_count, 16),
            )
        )
        page_start += page_count
    return tuple(result)


def _profile(
    entries: tuple[tuple[str, int, int, int], ...]
) -> Stage3StaticProfile:
    requests = _requests(entries)
    from llm.frontend.wafer_frontend.schema.common import ProfileKey

    return Stage3StaticProfile.create(
        key=ProfileKey(
            prefill_tokens=sum(item.prefill_tokens for item in requests),
            decode_tokens=sum(item.decode_tokens for item in requests),
            num_seqs=len(requests),
            context_sum=sum(item.context_tokens for item in requests),
            context_max=max(item.context_tokens for item in requests),
            kv_pages=sum(item.kv_span.page_count for item in requests),
            expert_load=None,
        ),
        requests=requests,
    )


def _profiles() -> tuple[Stage3StaticProfile, ...]:
    return (
        _profile((("r0", 8, 0, 8),)),
        _profile(
            tuple(
                (f"r{index}", 0, 1, context)
                for index, context in enumerate((4, 8, 12, 16, 20, 24, 28, 32))
            )
        ),
        _profile(
            (
                ("r0", 1, 0, 1),
                ("r1", 3, 0, 3),
                ("r2", 0, 1, 4),
                ("r3", 0, 1, 8),
                ("r4", 0, 1, 12),
                ("r5", 0, 1, 16),
            )
        ),
    )


def _template(
    profile: Stage3StaticProfile,
    tp: int,
    *,
    output: InferOutput = InferOutput.LOGITS,
) -> IR0Template:
    raw = valid_spec()
    raw["model"].update(  # type: ignore[index]
        V=32,
        H=16,
        I=32,
        NH=4,
        KVH=4,
        DH=4,
        rotary_dim=4,
        L=2,
        max_position_embeddings=128,
    )
    raw["parallel"]["instances"][0].update(  # type: ignore[index]
        role=profile.mode.value if profile.mode is not Stage3ProfileMode.MIXED else "both",
        tp=tp,
        sp=tp > 1,
    )
    raw["workload"]["infer"].update(  # type: ignore[index]
        output=output.value,
        profile={
            "prefill_tokens": profile.key.prefill_tokens,
            "decode_tokens": profile.key.decode_tokens,
            "num_seqs": profile.key.num_seqs,
            "context_sum": profile.key.context_sum,
            "context_max": profile.key.context_max,
            "kv_pages": profile.key.kv_pages,
            "expert_load": None,
        },
    )
    return build_ir0(
        from_data(ExperimentSpec, raw, path="spec"),
        exact_profiles=(profile,),
    )


class Stage3DenseInferenceOracleTest(unittest.TestCase):
    def test_exact_prefill_decode_and_mixed_profiles_reach_ir0(self) -> None:
        for profile in _profiles():
            with self.subTest(mode=profile.mode.value):
                graph = logical_expand(_template(profile, 1)).entries[0].graph
                attention = tuple(
                    node.workload
                    for node in graph.nodes
                    if node.kind is OpKind.ATTENTION
                )
                expected_mode = {
                    Stage3ProfileMode.PREFILL: AttentionMode.PREFILL,
                    Stage3ProfileMode.DECODE: AttentionMode.DECODE,
                    Stage3ProfileMode.MIXED: AttentionMode.MIXED,
                }[profile.mode]
                self.assertEqual(len(attention), 2)
                self.assertTrue(
                    all(
                        type(work) is AttentionWorkload
                        and work.exact_profile == profile
                        and work.mode is expected_mode
                        for work in attention
                    )
                )
                kv_states = tuple(
                    item
                    for item in graph.persistent_states
                    if item.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
                )
                self.assertEqual(len(kv_states), 4 * len(profile.requests))
                access_by_state = {
                    item.state_ref: item for item in graph.state_accesses
                }
                for request in profile.requests:
                    request_ref = f"{profile.id}:{request.request_ref}"
                    declarations = tuple(
                        item
                        for item in kv_states
                        if item.identity.request_ref == request_ref
                    )
                    self.assertEqual(len(declarations), 4)
                    for declaration in declarations:
                        access = access_by_state[declaration.id]
                        self.assertEqual(
                            declaration.shape,
                            (request.kv_span.capacity_tokens, 4, 4),
                        )
                        self.assertEqual(
                            access.mode,
                            StateAccessMode.READ_WRITE
                            if request.kv_read_tokens
                            else StateAccessMode.WRITE,
                        )
                        self.assertEqual(
                            (access.read_offset, access.read_shape),
                            ((0, 0, 0), (request.kv_read_tokens, 4, 4))
                            if request.kv_read_tokens
                            else (None, None),
                        )
                        self.assertEqual(
                            (access.write_offset, access.write_shape),
                            (
                                (
                                    request.context_tokens
                                    - request.query_tokens,
                                    0,
                                    0,
                                ),
                                (request.query_tokens, 4, 4),
                            ),
                        )

    def test_same_m_has_same_linear_work_but_distinct_attention_and_kv(self) -> None:
        prefill, decode, mixed = _profiles()
        oracles = tuple(
            build_stage3_dense_inference_oracle(
                _template(profile, 1), profile, tp_degree=1
            )
            for profile in (prefill, decode, mixed)
        )
        self.assertEqual(
            tuple(oracle.logical_work.gemm for oracle in oracles),
            (oracles[0].logical_work.gemm,) * 3,
        )
        self.assertEqual(
            tuple(
                (
                    oracle.logical_work.attention.query_key_pairs,
                    oracle.logical_work.attention.softmax_elements,
                    oracle.logical_work.attention.qk_matmul_flops,
                    oracle.logical_work.attention.vector_ops,
                    oracle.logical_work.attention.sfu_ops,
                )
                for oracle in oracles
            ),
            (
                (72, 288, 2304, 576, 288),
                (288, 1152, 9216, 2304, 1152),
                (94, 376, 3008, 752, 376),
            ),
        )
        self.assertEqual(
            tuple(
                (
                    oracle.kv.page_count,
                    oracle.kv.logical_read_bytes,
                    oracle.kv.logical_write_bytes,
                    oracle.kv.logical_reserved_bytes,
                )
                for oracle in oracles
            ),
            (
                (1, 0, 1024, 2048),
                (12, 18432, 1024, 24576),
                (6, 5120, 1024, 12288),
            ),
        )
        for oracle in oracles:
            self.assertEqual(
                (oracle.parameters.unique_elements, oracle.parameters.unique_bytes),
                (6224, 12448),
            )
            self.assertEqual(
                (oracle.graph.node_count, oracle.graph.collective_node_count),
                (25, 0),
            )

    def test_tp2_rank_work_collectives_and_kv_partition_are_exact(self) -> None:
        _prefill, decode, _mixed = _profiles()
        template = _template(decode, 2)
        oracle = build_stage3_dense_inference_oracle(
            template, decode, tp_degree=2
        )
        self.assertEqual(
            (oracle.parameters.placed_bytes, oracle.graph.node_count),
            (14656, 33),
        )
        self.assertEqual(
            (
                oracle.rank_work.attention.query_key_pairs,
                oracle.rank_work.attention.softmax_elements,
                oracle.rank_work.attention.qk_matmul_flops,
            ),
            (288, 576, 4608),
        )
        self.assertEqual(
            (
                oracle.kv.logical_read_bytes,
                oracle.kv.rank_read_bytes,
                oracle.kv.logical_reserved_bytes,
                oracle.kv.rank_reserved_bytes,
            ),
            (18432, 9216, 24576, 12288),
        )
        self.assertEqual(
            (
                oracle.collectives.all_gather.node_count,
                oracle.collectives.all_gather.rank_payload_bytes_total,
                oracle.collectives.all_gather.group_payload_bytes_total,
            ),
            (4, 512, 1024),
        )

    def test_strict_roundtrip_determinism_and_template_cross_gate(self) -> None:
        for profile in _profiles():
            with self.subTest(mode=profile.mode.value):
                template = _template(profile, 1)
                oracle = build_stage3_dense_inference_oracle(
                    template, profile, tp_degree=1
                )
                oracle.validate_against_template(template)
                self.assertEqual(
                    loads_dataclass(
                        Stage3DenseInferenceOracle,
                        canonical_json(oracle),
                        path="oracle",
                    ),
                    oracle,
                )
                self.assertEqual(
                    build_stage3_dense_inference_oracle(
                        template, profile, tp_degree=1
                    ),
                    oracle,
                )

    def test_fail_closed_profile_role_tp_and_restable_metric_tamper(self) -> None:
        prefill, decode, _mixed = _profiles()
        template = _template(decode, 1)
        with self.assertRaisesRegex(SchemaError, "absent from template"):
            build_stage3_dense_inference_oracle(
                template, prefill, tp_degree=1
            )
        wrong_role = IR0Template.create(
            job=template.job,
            model=template.model,
            instance=replace(template.instance, role=LogicalRole.PREFILL),
            sequence_parallel=template.sequence_parallel,
            layer=template.layer,
            infer_output=template.infer_output,
            profiles=template.profiles,
        )
        with self.assertRaisesRegex(SchemaError, "incompatible with instance role"):
            build_stage3_dense_inference_oracle(
                wrong_role, decode, tp_degree=1
            )
        with self.assertRaisesRegex(SchemaError, "must equal template TP"):
            build_stage3_dense_inference_oracle(
                template, decode, tp_degree=2
            )
        oracle = build_stage3_dense_inference_oracle(
            template, decode, tp_degree=1
        )
        tampered = Stage3DenseInferenceOracle.create(
            **{
                **oracle._semantic_key(),
                "kv": replace(
                    oracle.kv,
                    logical_read_bytes=oracle.kv.logical_read_bytes + 64,
                    rank_read_bytes=oracle.kv.rank_read_bytes + 64,
                ),
            }
        )
        with self.assertRaisesRegex(SchemaError, "source template/profile"):
            tampered.validate_against_template(template)
        with self.assertRaisesRegex(UnsupportedFeatureError, "TP-sharded greedy"):
            build_stage3_dense_inference_oracle(
                _template(decode, 2, output=InferOutput.GREEDY_SAMPLE),
                decode,
                tp_degree=2,
            )


if __name__ == "__main__":
    unittest.main()
