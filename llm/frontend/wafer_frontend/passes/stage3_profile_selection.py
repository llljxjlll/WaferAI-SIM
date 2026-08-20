"""Dominance-only selection for exact Stage 3 static profiles."""

from __future__ import annotations

from ..errors import SchemaError, StageNotImplementedError, UnsupportedFeatureError
from ..schema.stage3_profile import (
    Stage3ProfileCapacity,
    Stage3ProfileFallback,
    Stage3ProfileSelection,
    Stage3StaticProfile,
)


_CAPACITY_FIELDS = (
    "prefill_tokens",
    "decode_tokens",
    "num_seqs",
    "context_sum",
    "context_max",
    "kv_pages",
    "query_key_pairs",
    "kv_read_tokens",
    "kv_write_tokens",
)


def _covers(capacity: Stage3ProfileCapacity, request: Stage3ProfileCapacity) -> bool:
    return capacity.covers(request)


def _strictly_smaller(
    left: Stage3ProfileCapacity, right: Stage3ProfileCapacity
) -> bool:
    return (
        left.mode is right.mode
        and all(
            getattr(left, field_name) <= getattr(right, field_name)
            for field_name in _CAPACITY_FIELDS
        )
        and any(
            getattr(left, field_name) < getattr(right, field_name)
            for field_name in _CAPACITY_FIELDS
        )
    )


def select_stage3_profile(
    profiles: tuple[Stage3StaticProfile, ...],
    request: Stage3StaticProfile,
    *,
    fallback: Stage3ProfileFallback = Stage3ProfileFallback.REJECT,
) -> Stage3ProfileSelection:
    """Select the unique Pareto-minimal dominating profile.

    No scalar distance or nearest-bucket heuristic is used. Incomparable
    minimal candidates are rejected because choosing between them would require
    an explicit, versioned cost policy.
    """

    if type(fallback) is not Stage3ProfileFallback:
        raise SchemaError("must be a Stage3ProfileFallback", path="fallback")
    request.validate("request")
    if not profiles:
        raise SchemaError("must contain at least one profile", path="profiles")
    by_id: dict[str, Stage3StaticProfile] = {}
    for index, profile in enumerate(profiles):
        if type(profile) is not Stage3StaticProfile:
            raise SchemaError(
                "must be a Stage3StaticProfile", path=f"profiles[{index}]"
            )
        profile.validate(f"profiles[{index}]")
        if profile.id in by_id:
            raise SchemaError(
                "contains a duplicate profile id", path=f"profiles[{index}].id"
            )
        by_id[profile.id] = profile
    considered = tuple(sorted(by_id))
    exact = by_id.get(request.id)
    if exact is not None:
        selected = exact
    else:
        candidates = tuple(
            profile
            for profile in by_id.values()
            if _covers(profile.capacity, request.capacity)
        )
        minimal = tuple(
            candidate
            for candidate in candidates
            if not any(
                other.id != candidate.id
                and _strictly_smaller(other.capacity, candidate.capacity)
                for other in candidates
            )
        )
        if not minimal:
            if fallback is Stage3ProfileFallback.REJECT:
                raise UnsupportedFeatureError(
                    "no Stage 3 profile dominates the requested exact shape",
                    path="request.key",
                )
            raise StageNotImplementedError(
                f"declared {fallback.value!r} profile fallback is unavailable",
                path="fallback",
            )
        if len(minimal) != 1:
            raise SchemaError(
                "dominating profiles have no unique Pareto-minimal candidate",
                path="profiles",
            )
        selected = minimal[0]
    return Stage3ProfileSelection.create(
        request_profile_id=request.id,
        selected_profile_id=selected.id,
        exact_match=selected.id == request.id,
        fallback=Stage3ProfileFallback.REJECT,
        request_capacity=request.capacity,
        selected_capacity=selected.capacity,
        considered_profile_ids=considered,
    )


__all__ = ["select_stage3_profile"]
