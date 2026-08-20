from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    StageNotImplementedError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes import select_stage3_profile
from llm.frontend.wafer_frontend.schema.common import ProfileKey
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.stage3_profile import (
    KvPageSpan,
    Stage3ProfileFallback,
    Stage3ProfileMode,
    Stage3ProfileSelection,
    Stage3StaticProfile,
    StaticRequestShape,
)


def _request(
    request_ref: str,
    *,
    prefill: int = 0,
    decode: int = 0,
    context: int,
    page_start: int,
    page_size: int = 16,
) -> StaticRequestShape:
    pages = (context + page_size - 1) // page_size
    return StaticRequestShape(
        request_ref=request_ref,
        prefill_tokens=prefill,
        decode_tokens=decode,
        context_tokens=context,
        kv_span=KvPageSpan(page_start, pages, page_size),
    )


def _profile(*requests: StaticRequestShape) -> Stage3StaticProfile:
    key = ProfileKey(
        prefill_tokens=sum(item.prefill_tokens for item in requests),
        decode_tokens=sum(item.decode_tokens for item in requests),
        num_seqs=len(requests),
        context_sum=sum(item.context_tokens for item in requests),
        context_max=max(item.context_tokens for item in requests),
        kv_pages=sum(item.kv_span.page_count for item in requests),
        expert_load=None,
    )
    return Stage3StaticProfile.create(key=key, requests=tuple(requests))


class Stage3ProfileSelectionTest(unittest.TestCase):
    def test_prefill_decode_mixed_and_ragged_metrics_are_exact(self) -> None:
        prefill = _profile(_request("r0", prefill=8, context=8, page_start=0))
        decode = _profile(
            _request("r0", decode=1, context=4, page_start=0),
            _request("r1", decode=1, context=8, page_start=1),
        )
        mixed = _profile(
            _request("r0", prefill=3, context=3, page_start=0),
            _request("r1", decode=1, context=4, page_start=1),
            _request("r2", decode=1, context=9, page_start=2),
        )
        ragged = _profile(
            _request("r0", prefill=3, context=3, page_start=0),
            _request("r1", prefill=5, context=5, page_start=1),
        )
        self.assertEqual(
            (
                prefill.mode,
                prefill.capacity.query_key_pairs,
                prefill.capacity.kv_read_tokens,
            ),
            (Stage3ProfileMode.PREFILL, 36, 0),
        )
        self.assertEqual(
            (
                decode.mode,
                decode.capacity.query_key_pairs,
                decode.capacity.kv_read_tokens,
            ),
            (Stage3ProfileMode.DECODE, 12, 12),
        )
        self.assertEqual(
            (
                mixed.mode,
                mixed.capacity.query_key_pairs,
                mixed.capacity.kv_read_tokens,
                mixed.capacity.kv_write_tokens,
            ),
            (Stage3ProfileMode.MIXED, 19, 13, 5),
        )
        self.assertEqual(
            (
                ragged.capacity.query_key_pairs,
                ragged.key.context_sum,
                ragged.key.context_max,
            ),
            (21, 8, 5),
        )
        for profile in (prefill, decode, mixed, ragged):
            self.assertEqual(
                loads_dataclass(
                    Stage3StaticProfile, canonical_json(profile), path="profile"
                ),
                profile,
            )

    def test_dominance_exact_unique_minimum_and_roundtrip(self) -> None:
        request = _profile(
            _request("r0", decode=1, context=4, page_start=0),
            _request("r1", decode=1, context=8, page_start=1),
        )
        larger = _profile(
            _request("r0", decode=1, context=8, page_start=0),
            _request("r1", decode=1, context=12, page_start=1),
            _request("r2", decode=1, context=12, page_start=2),
        )
        largest = _profile(
            _request("r0", decode=1, context=16, page_start=0),
            _request("r1", decode=1, context=16, page_start=1),
            _request("r2", decode=1, context=16, page_start=2),
            _request("r3", decode=1, context=16, page_start=3),
        )
        exact = select_stage3_profile((largest, request, larger), request)
        self.assertTrue(exact.exact_match)
        self.assertEqual(exact.selected_profile_id, request.id)
        dominated = select_stage3_profile((largest, larger), request)
        self.assertFalse(dominated.exact_match)
        self.assertEqual(dominated.selected_profile_id, larger.id)
        self.assertEqual(
            loads_dataclass(
                Stage3ProfileSelection,
                canonical_json(dominated),
                path="selection",
            ),
            dominated,
        )

    def test_fail_closed_shape_page_and_selection_negatives(self) -> None:
        base = _profile(_request("r0", decode=1, context=8, page_start=0))
        with self.assertRaisesRegex(SchemaError, "exactly one"):
            replace(base.requests[0], prefill_tokens=1).validate("request")
        with self.assertRaisesRegex(SchemaError, "minimally cover"):
            replace(
                base.requests[0],
                kv_span=replace(base.requests[0].kv_span, page_count=2),
            ).validate("request")
        overlap = _profile(
            _request("r0", decode=1, context=8, page_start=0),
            _request("r1", decode=1, context=8, page_start=1),
        )
        with self.assertRaisesRegex(SchemaError, "must not overlap"):
            replace(
                overlap,
                requests=(
                    replace(
                        overlap.requests[0],
                        kv_span=KvPageSpan(0, 2, 4),
                        context_tokens=8,
                    ),
                    replace(overlap.requests[1], kv_span=KvPageSpan(1, 2, 4)),
                ),
            ).validate("profile")
        with self.assertRaisesRegex(SchemaError, "one page_size_tokens"):
            replace(
                overlap,
                requests=(
                    overlap.requests[0],
                    replace(
                        overlap.requests[1],
                        kv_span=KvPageSpan(2, 2, 4),
                    ),
                ),
            ).validate("profile")
        too_small = _profile(_request("r0", decode=1, context=4, page_start=0))
        with self.assertRaisesRegex(UnsupportedFeatureError, "no Stage 3 profile"):
            select_stage3_profile((too_small,), base)
        with self.assertRaisesRegex(StageNotImplementedError, "recompile"):
            select_stage3_profile(
                (too_small,), base, fallback=Stage3ProfileFallback.RECOMPILE
            )
        selected = select_stage3_profile((base,), base)
        with self.assertRaisesRegex(SchemaError, "must dominate"):
            Stage3ProfileSelection.create(
                **{
                    **selected._semantic_key(),
                    "exact_match": False,
                    "selected_profile_id": too_small.id,
                    "selected_capacity": replace(
                        selected.selected_capacity,
                        query_key_pairs=selected.request_capacity.query_key_pairs - 1,
                    ),
                    "considered_profile_ids": tuple(sorted((base.id, too_small.id))),
                }
            )

    def test_incomparable_minima_and_mode_substitution_are_rejected(self) -> None:
        request = _profile(_request("r0", decode=1, context=4, page_start=0))
        higher_context = _profile(
            _request("r0", decode=1, context=8, page_start=0)
        )
        more_sequences = _profile(
            _request("r0", decode=1, context=4, page_start=0),
            _request("r1", decode=1, context=4, page_start=1),
        )
        with self.assertRaisesRegex(SchemaError, "no unique Pareto"):
            select_stage3_profile((higher_context, more_sequences), request)
        prefill = _profile(_request("r0", prefill=8, context=8, page_start=0))
        with self.assertRaisesRegex(UnsupportedFeatureError, "no Stage 3 profile"):
            select_stage3_profile((prefill,), request)


if __name__ == "__main__":
    unittest.main()
