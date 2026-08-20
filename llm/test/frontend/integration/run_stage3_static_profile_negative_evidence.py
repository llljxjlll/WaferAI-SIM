#!/usr/bin/env python3
"""Execute reviewed Stage3 static-profile fail-closed witnesses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from typing import Callable


_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.errors import (  # noqa: E402
    FrontendError,
    SchemaError,
    StageNotImplementedError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    select_stage3_profile,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    ProfileKey,
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (  # noqa: E402
    KvPageSpan,
    Stage3ProfileFallback,
    Stage3StaticProfile,
    StaticRequestShape,
)
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (  # noqa: E402
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
)


_SCHEMA_VERSION = (
    "wafer_frontend.stage3_static_profile_negative_evidence/v1alpha1"
)
_PRODUCER_PASS = "stage3_static_profile_negative_runner"
_COMMAND = (
    "python3 -B llm/test/frontend/integration/"
    "run_stage3_static_profile_negative_evidence.py [--output <path>]"
)
_SOURCE_PATHS = (
    "llm/frontend/wafer_frontend/passes/stage3_profile_selection.py",
    "llm/frontend/wafer_frontend/schema/stage3_profile.py",
    "llm/test/frontend/integration/"
    "run_stage3_static_profile_negative_evidence.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        requests=tuple(requests),
    )


def _bindings(
    *profiles: Stage3StaticProfile,
) -> dict[str, tuple[dict[str, str], ...]]:
    result = {
        "tools": (
            {"name": "python", "sha256": _sha256(Path(sys.executable))},
        ),
        "sources": tuple(
            {"path": path, "sha256": _sha256(_ROOT / path)}
            for path in _SOURCE_PATHS
        ),
        "inputs": tuple(
            {
                "name": f"profile.{index}",
                "sha256": canonical_digest(profile),
            }
            for index, profile in enumerate(profiles)
        ),
    }
    if not result["inputs"]:
        result["inputs"] = (
            {
                "name": "empty-profile-set",
                "sha256": canonical_digest(()),
            },
        )
    return result


def _expect(
    *,
    key: str,
    validator: str,
    expected_type: type[FrontendError],
    expected_message: str,
    mutation: Callable[[], None],
    bindings: dict[str, tuple[dict[str, str], ...]],
) -> dict[str, object]:
    try:
        mutation()
    except FrontendError as error:
        observed = str(error)
        if type(error) is not expected_type or observed != expected_message:
            raise RuntimeError(
                f"{key}: expected {expected_type.__name__} "
                f"{expected_message!r}, observed {type(error).__name__} "
                f"{observed!r}"
            ) from error
        return {
            "bindings": bindings,
            "expected_error": expected_type.__name__,
            "expected_message": expected_message,
            "key": key,
            "observed_error": observed,
            "passed": True,
            "validator": validator,
        }
    except Exception as error:  # pragma: no cover
        raise RuntimeError(
            f"{key}: expected {expected_type.__name__}, observed "
            f"{type(error).__name__}: {error}"
        ) from error
    raise RuntimeError(f"{key}: mutation was incorrectly accepted")


def _witnesses() -> tuple[dict[str, object], ...]:
    request = _profile(_request("r0", decode=1, context=8, page_start=0))
    too_small = _profile(
        _request("r0", decode=1, context=4, page_start=0)
    )
    prefill = _profile(
        _request("r0", prefill=8, context=8, page_start=0)
    )
    higher_context = _profile(
        _request("r0", decode=1, context=16, page_start=0)
    )
    more_sequences = _profile(
        _request("r0", decode=1, context=8, page_start=0),
        _request("r1", decode=1, context=8, page_start=1),
    )
    overlap = _profile(
        _request("r0", decode=1, context=8, page_start=0),
        _request("r1", decode=1, context=8, page_start=1),
    )
    overlapping = replace(
        overlap,
        requests=(
            replace(
                overlap.requests[0],
                kv_span=KvPageSpan(0, 2, 4),
            ),
            replace(
                overlap.requests[1],
                kv_span=KvPageSpan(1, 2, 4),
            ),
        ),
    )
    overprovisioned = replace(
        request.requests[0],
        kv_span=replace(request.requests[0].kv_span, page_count=2),
    )
    witnesses = (
        _expect(
            key="duplicate_profile_id",
            validator="select_stage3_profile",
            expected_type=SchemaError,
            expected_message=(
                "schema_error at profiles[1].id: "
                "contains a duplicate profile id"
            ),
            mutation=lambda: select_stage3_profile(
                (too_small, too_small), request
            ),
            bindings=_bindings(too_small, request),
        ),
        _expect(
            key="empty_profile_set",
            validator="select_stage3_profile",
            expected_type=SchemaError,
            expected_message=(
                "schema_error at profiles: must contain at least one profile"
            ),
            mutation=lambda: select_stage3_profile((), request),
            bindings=_bindings(request),
        ),
        _expect(
            key="fallback_eager_unavailable",
            validator="select_stage3_profile",
            expected_type=StageNotImplementedError,
            expected_message=(
                "stage_not_implemented at fallback: declared 'eager' "
                "profile fallback is unavailable"
            ),
            mutation=lambda: select_stage3_profile(
                (too_small,), request, fallback=Stage3ProfileFallback.EAGER
            ),
            bindings=_bindings(too_small, request),
        ),
        _expect(
            key="fallback_recompile_unavailable",
            validator="select_stage3_profile",
            expected_type=StageNotImplementedError,
            expected_message=(
                "stage_not_implemented at fallback: declared 'recompile' "
                "profile fallback is unavailable"
            ),
            mutation=lambda: select_stage3_profile(
                (too_small,), request,
                fallback=Stage3ProfileFallback.RECOMPILE,
            ),
            bindings=_bindings(too_small, request),
        ),
        _expect(
            key="incomparable_minima",
            validator="select_stage3_profile",
            expected_type=SchemaError,
            expected_message=(
                "schema_error at profiles: dominating profiles have no unique "
                "Pareto-minimal candidate"
            ),
            mutation=lambda: select_stage3_profile(
                (higher_context, more_sequences), too_small
            ),
            bindings=_bindings(higher_context, more_sequences, too_small),
        ),
        _expect(
            key="kv_page_overprovision",
            validator="StaticRequestShape.validate",
            expected_type=SchemaError,
            expected_message=(
                "schema_error at request.kv_span.page_count: "
                "must minimally cover context_tokens with 1 pages"
            ),
            mutation=lambda: overprovisioned.validate("request"),
            bindings=_bindings(request),
        ),
        _expect(
            key="kv_span_overlap",
            validator="Stage3StaticProfile.validate",
            expected_type=SchemaError,
            expected_message=(
                "schema_error at profile.requests[1].kv_span: "
                "KV page spans must not overlap"
            ),
            mutation=lambda: overlapping.validate("profile"),
            bindings=_bindings(overlap),
        ),
        _expect(
            key="mode_substitution",
            validator="select_stage3_profile",
            expected_type=UnsupportedFeatureError,
            expected_message=(
                "unsupported_feature at request.key: no Stage 3 profile "
                "dominates the requested exact shape"
            ),
            mutation=lambda: select_stage3_profile((prefill,), request),
            bindings=_bindings(prefill, request),
        ),
        _expect(
            key="no_dominating_profile",
            validator="select_stage3_profile",
            expected_type=UnsupportedFeatureError,
            expected_message=(
                "unsupported_feature at request.key: no Stage 3 profile "
                "dominates the requested exact shape"
            ),
            mutation=lambda: select_stage3_profile((too_small,), request),
            bindings=_bindings(too_small, request),
        ),
    )
    return tuple(sorted(witnesses, key=lambda item: str(item["key"])))


def _validate_evidence(evidence: dict[str, object]) -> None:
    if evidence.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError("negative evidence schema version changed")
    witnesses = evidence.get("witnesses")
    if type(witnesses) not in (list, tuple) or len(witnesses) != 9:
        raise RuntimeError("negative evidence must contain exactly nine witnesses")
    expected_keys = (
        "duplicate_profile_id",
        "empty_profile_set",
        "fallback_eager_unavailable",
        "fallback_recompile_unavailable",
        "incomparable_minima",
        "kv_page_overprovision",
        "kv_span_overlap",
        "mode_substitution",
        "no_dominating_profile",
    )
    if tuple(item.get("key") for item in witnesses) != expected_keys:
        raise RuntimeError("negative witness key set/order changed")
    for witness in witnesses:
        bindings = witness.get("bindings")
        if (
            witness.get("passed") is not True
            or type(bindings) is not dict
            or any(
                not bindings.get(name)
                for name in ("tools", "sources", "inputs")
            )
        ):
            raise RuntimeError(f"negative witness is incomplete: {witness}")
        for group in bindings.values():
            for item in group:
                digest = item.get("sha256")
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise RuntimeError(
                        f"negative witness digest is invalid: {item}"
                    )
    expected_sources = tuple(
        {"path": path, "sha256": _sha256(_ROOT / path)}
        for path in _SOURCE_PATHS
    )
    if canonical_json(evidence.get("source_digests")) != canonical_json(
        expected_sources
    ):
        raise RuntimeError("negative evidence source digests changed")
    if evidence.get("producer_pass") != _PRODUCER_PASS:
        raise RuntimeError("negative evidence producer changed")
    if evidence.get("baseline_epoch") != STAGE3_STATIC_PROFILE_BASELINE_EPOCH:
        raise RuntimeError("negative evidence baseline epoch changed")
    if evidence.get("command") != _COMMAND:
        raise RuntimeError("negative evidence command changed")
    expected_witnesses = _witnesses()
    if canonical_json(witnesses) != canonical_json(expected_witnesses):
        raise RuntimeError("negative evidence witness payload changed")
    semantic_key = {
        "baseline_epoch": STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
        "command": _COMMAND,
        "source_digests": expected_sources,
        "witnesses": expected_witnesses,
    }
    expected_id = stable_artifact_id(
        "stage3_static_profile_negative_evidence",
        semantic_key,
        schema_version=_SCHEMA_VERSION,
    )
    if evidence.get("id") != expected_id:
        raise RuntimeError("negative evidence stable id changed")


def build_negative_evidence() -> dict[str, object]:
    semantic_key = {
        "baseline_epoch": STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
        "command": _COMMAND,
        "source_digests": tuple(
            {"path": path, "sha256": _sha256(_ROOT / path)}
            for path in _SOURCE_PATHS
        ),
        "witnesses": _witnesses(),
    }
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "producer_pass": _PRODUCER_PASS,
        "id": stable_artifact_id(
            "stage3_static_profile_negative_evidence",
            semantic_key,
            schema_version=_SCHEMA_VERSION,
        ),
        **semantic_key,
    }
    _validate_evidence(evidence)
    return evidence


def _write_new(path: Path, evidence: dict[str, object]) -> None:
    if path.suffix != ".json":
        raise RuntimeError("checked negative evidence output must be JSON, not NPUP")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite/symlink checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(evidence) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.absolute()
        if args.output.suffix != ".json":
            raise RuntimeError(
                "checked negative evidence output must be JSON, not NPUP"
            )
        if args.output.exists() or args.output.is_symlink():
            raise RuntimeError(
                f"refusing to overwrite/symlink checked evidence: {args.output}"
            )
    first = build_negative_evidence()
    second = build_negative_evidence()
    first_bytes = (canonical_json(first) + "\n").encode("utf-8")
    second_bytes = (canonical_json(second) + "\n").encode("utf-8")
    if first_bytes != second_bytes:
        raise RuntimeError("negative evidence repeats are not byte-identical")
    if args.output is not None:
        _write_new(args.output, first)
    print(
        "[STAGE3 NEGATIVE] PASS: "
        f"witnesses={len(first['witnesses'])} deterministic=1 id={first['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
