"""Exact static request shapes and profile-selection evidence for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import ProfileKey, stable_artifact_id, validate_nonempty, validate_uint64


STAGE3_STATIC_PROFILE_SCHEMA_VERSION = (
    "wafer_frontend.stage3_static_profile/v1alpha1"
)
STAGE3_PROFILE_SELECTION_SCHEMA_VERSION = (
    "wafer_frontend.stage3_profile_selection/v1alpha1"
)


class Stage3ProfileMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class Stage3ProfileFallback(str, Enum):
    REJECT = "reject"
    RECOMPILE = "recompile"
    EAGER = "eager"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


@dataclass(frozen=True, slots=True)
class KvPageSpan:
    """One request's exact, disjoint paged-KV capacity span."""

    page_start: int
    page_count: int
    page_size_tokens: int

    @property
    def page_end(self) -> int:
        return self.page_start + self.page_count

    @property
    def capacity_tokens(self) -> int:
        return self.page_count * self.page_size_tokens

    def validate(self, path: str = "kv_page_span") -> None:
        validate_uint64(self.page_start, f"{path}.page_start")
        _positive(self.page_count, f"{path}.page_count")
        _positive(self.page_size_tokens, f"{path}.page_size_tokens")
        validate_uint64(self.page_end, f"{path}.derived.page_end")
        validate_uint64(self.capacity_tokens, f"{path}.derived.capacity_tokens")


@dataclass(frozen=True, slots=True)
class StaticRequestShape:
    """Exact per-request causal query and KV shape for one static iteration.

    ``context_tokens`` is the request's context length after the iteration. A
    query chunk of ``q`` tokens therefore performs
    ``q * (2 * context_tokens - q + 1) / 2`` causal query-key pairs. This
    representation distinguishes ragged requests even when aggregate M is the
    same.
    """

    request_ref: str
    prefill_tokens: int
    decode_tokens: int
    context_tokens: int
    kv_span: KvPageSpan

    @property
    def query_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def mode(self) -> Stage3ProfileMode:
        return (
            Stage3ProfileMode.PREFILL
            if self.prefill_tokens
            else Stage3ProfileMode.DECODE
        )

    @property
    def query_key_pairs(self) -> int:
        q = self.query_tokens
        return q * (2 * self.context_tokens - q + 1) // 2

    @property
    def kv_read_tokens(self) -> int:
        if self.decode_tokens:
            return self.context_tokens
        return self.context_tokens - self.prefill_tokens

    @property
    def kv_write_tokens(self) -> int:
        return self.query_tokens

    def validate(self, path: str = "request") -> None:
        validate_nonempty(self.request_ref, f"{path}.request_ref")
        validate_uint64(self.prefill_tokens, f"{path}.prefill_tokens")
        validate_uint64(self.decode_tokens, f"{path}.decode_tokens")
        _positive(self.context_tokens, f"{path}.context_tokens")
        if bool(self.prefill_tokens) == bool(self.decode_tokens):
            raise SchemaError(
                "exactly one of prefill_tokens/decode_tokens must be non-zero",
                path=path,
            )
        _positive(self.query_tokens, f"{path}.derived.query_tokens")
        if self.context_tokens < self.query_tokens:
            raise SchemaError(
                "must be at least the query-token count",
                path=f"{path}.context_tokens",
            )
        _positive(self.query_key_pairs, f"{path}.derived.query_key_pairs")
        validate_uint64(self.kv_read_tokens, f"{path}.derived.kv_read_tokens")
        validate_uint64(self.kv_write_tokens, f"{path}.derived.kv_write_tokens")
        if type(self.kv_span) is not KvPageSpan:
            raise SchemaError("must be a KvPageSpan", path=f"{path}.kv_span")
        self.kv_span.validate(f"{path}.kv_span")
        expected_pages = (
            self.context_tokens + self.kv_span.page_size_tokens - 1
        ) // self.kv_span.page_size_tokens
        if self.kv_span.page_count != expected_pages:
            raise SchemaError(
                f"must minimally cover context_tokens with {expected_pages} pages",
                path=f"{path}.kv_span.page_count",
            )


@dataclass(frozen=True, slots=True)
class Stage3ProfileCapacity:
    mode: Stage3ProfileMode
    prefill_tokens: int
    decode_tokens: int
    num_seqs: int
    context_sum: int
    context_max: int
    kv_pages: int
    query_key_pairs: int
    kv_read_tokens: int
    kv_write_tokens: int

    def covers(self, request: "Stage3ProfileCapacity") -> bool:
        fields = (
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
        return self.mode is request.mode and all(
            getattr(self, field_name) >= getattr(request, field_name)
            for field_name in fields
        )

    def validate(self, path: str = "capacity") -> None:
        if type(self.mode) is not Stage3ProfileMode:
            raise SchemaError("must be a Stage3ProfileMode", path=f"{path}.mode")
        for field_name in (
            "prefill_tokens",
            "decode_tokens",
            "num_seqs",
            "context_sum",
            "context_max",
            "kv_pages",
            "query_key_pairs",
            "kv_read_tokens",
            "kv_write_tokens",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.num_seqs == 0 or self.query_key_pairs == 0:
            raise SchemaError("must describe non-empty work", path=path)
        if self.kv_pages == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.kv_pages")
        if self.context_max > self.context_sum:
            raise SchemaError(
                "must not exceed context_sum", path=f"{path}.context_max"
            )
        if self.prefill_tokens + self.decode_tokens < self.num_seqs:
            raise SchemaError(
                "query-token count must be at least num_seqs", path=path
            )
        if self.kv_write_tokens != self.prefill_tokens + self.decode_tokens:
            raise SchemaError(
                "must equal prefill_tokens + decode_tokens",
                path=f"{path}.kv_write_tokens",
            )
        expected_mode = (
            Stage3ProfileMode.MIXED
            if self.prefill_tokens and self.decode_tokens
            else (
                Stage3ProfileMode.PREFILL
                if self.prefill_tokens
                else Stage3ProfileMode.DECODE
            )
        )
        if self.mode is not expected_mode:
            raise SchemaError(
                f"must equal token-derived mode {expected_mode.value!r}",
                path=f"{path}.mode",
            )


@dataclass(frozen=True, slots=True)
class Stage3StaticProfile:
    schema_version: str
    producer_pass: str
    id: str
    key: ProfileKey
    requests: tuple[StaticRequestShape, ...]

    @classmethod
    def create(
        cls,
        *,
        key: ProfileKey,
        requests: tuple[StaticRequestShape, ...],
    ) -> "Stage3StaticProfile":
        semantic_key = {"key": key, "requests": requests}
        result = cls(
            schema_version=STAGE3_STATIC_PROFILE_SCHEMA_VERSION,
            producer_pass="stage3_static_profile",
            id=stable_artifact_id(
                "stage3_static_profile",
                semantic_key,
                schema_version=STAGE3_STATIC_PROFILE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    @property
    def mode(self) -> Stage3ProfileMode:
        has_prefill = any(item.prefill_tokens for item in self.requests)
        has_decode = any(item.decode_tokens for item in self.requests)
        if has_prefill and has_decode:
            return Stage3ProfileMode.MIXED
        return Stage3ProfileMode.PREFILL if has_prefill else Stage3ProfileMode.DECODE

    @property
    def capacity(self) -> Stage3ProfileCapacity:
        return Stage3ProfileCapacity(
            mode=self.mode,
            prefill_tokens=self.key.prefill_tokens,
            decode_tokens=self.key.decode_tokens,
            num_seqs=self.key.num_seqs,
            context_sum=self.key.context_sum,
            context_max=self.key.context_max,
            kv_pages=self.key.kv_pages,
            query_key_pairs=sum(item.query_key_pairs for item in self.requests),
            kv_read_tokens=sum(item.kv_read_tokens for item in self.requests),
            kv_write_tokens=sum(item.kv_write_tokens for item in self.requests),
        )

    def _semantic_key(self) -> dict[str, object]:
        return {"key": self.key, "requests": self.requests}

    def validate(self, path: str = "stage3_profile") -> None:
        if self.schema_version != STAGE3_STATIC_PROFILE_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "stage3_static_profile":
            raise SchemaError(
                "must be 'stage3_static_profile'", path=f"{path}.producer_pass"
            )
        self.key.validate(f"{path}.key")
        if self.key.expert_load is not None:
            raise SchemaError(
                "must be null for a Dense profile", path=f"{path}.key.expert_load"
            )
        if not self.requests:
            raise SchemaError("must contain at least one request", path=f"{path}.requests")
        previous_ref: str | None = None
        page_size_tokens: int | None = None
        spans: list[tuple[int, int, str]] = []
        for index, request in enumerate(self.requests):
            request_path = f"{path}.requests[{index}]"
            if type(request) is not StaticRequestShape:
                raise SchemaError("must be a StaticRequestShape", path=request_path)
            request.validate(request_path)
            if page_size_tokens is None:
                page_size_tokens = request.kv_span.page_size_tokens
            elif request.kv_span.page_size_tokens != page_size_tokens:
                raise SchemaError(
                    "all request spans must use one page_size_tokens",
                    path=f"{request_path}.kv_span.page_size_tokens",
                )
            if previous_ref is not None and request.request_ref <= previous_ref:
                raise SchemaError(
                    "request_ref values must be strictly increasing",
                    path=f"{request_path}.request_ref",
                )
            previous_ref = request.request_ref
            spans.append(
                (
                    request.kv_span.page_start,
                    request.kv_span.page_end,
                    request_path,
                )
            )
        previous_end: int | None = None
        for start, end, request_path in sorted(spans):
            if previous_end is not None and start < previous_end:
                raise SchemaError(
                    "KV page spans must not overlap", path=f"{request_path}.kv_span"
                )
            previous_end = end
        expected = {
            "prefill_tokens": sum(item.prefill_tokens for item in self.requests),
            "decode_tokens": sum(item.decode_tokens for item in self.requests),
            "num_seqs": len(self.requests),
            "context_sum": sum(item.context_tokens for item in self.requests),
            "context_max": max(item.context_tokens for item in self.requests),
            "kv_pages": sum(item.kv_span.page_count for item in self.requests),
        }
        for field_name, value in expected.items():
            if getattr(self.key, field_name) != value:
                raise SchemaError(
                    f"must equal the request-derived value {value}",
                    path=f"{path}.key.{field_name}",
                )
        self.capacity.validate(f"{path}.capacity")
        expected_id = stable_artifact_id(
            "stage3_static_profile",
            self._semantic_key(),
            schema_version=STAGE3_STATIC_PROFILE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class Stage3ProfileSelection:
    schema_version: str
    producer_pass: str
    id: str
    request_profile_id: str
    selected_profile_id: str
    exact_match: bool
    fallback: Stage3ProfileFallback
    request_capacity: Stage3ProfileCapacity
    selected_capacity: Stage3ProfileCapacity
    considered_profile_ids: tuple[str, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage3ProfileSelection":
        result = cls(
            schema_version=STAGE3_PROFILE_SELECTION_SCHEMA_VERSION,
            producer_pass="stage3_profile_selection",
            id=stable_artifact_id(
                "stage3_profile_selection",
                semantic_key,
                schema_version=STAGE3_PROFILE_SELECTION_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "request_profile_id": self.request_profile_id,
            "selected_profile_id": self.selected_profile_id,
            "exact_match": self.exact_match,
            "fallback": self.fallback,
            "request_capacity": self.request_capacity,
            "selected_capacity": self.selected_capacity,
            "considered_profile_ids": self.considered_profile_ids,
        }

    def validate(self, path: str = "profile_selection") -> None:
        if self.schema_version != STAGE3_PROFILE_SELECTION_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {self.schema_version!r}",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "stage3_profile_selection":
            raise SchemaError(
                "must be 'stage3_profile_selection'",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.request_profile_id, f"{path}.request_profile_id")
        validate_nonempty(self.selected_profile_id, f"{path}.selected_profile_id")
        if type(self.exact_match) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.exact_match")
        if type(self.fallback) is not Stage3ProfileFallback:
            raise SchemaError(
                "must be a Stage3ProfileFallback", path=f"{path}.fallback"
            )
        if self.fallback is not Stage3ProfileFallback.REJECT:
            raise SchemaError(
                "successful selection must not claim an unavailable fallback",
                path=f"{path}.fallback",
            )
        self.request_capacity.validate(f"{path}.request_capacity")
        self.selected_capacity.validate(f"{path}.selected_capacity")
        if self.request_capacity.mode is not self.selected_capacity.mode:
            raise SchemaError(
                "selected profile mode must equal request mode",
                path=f"{path}.selected_capacity.mode",
            )
        if not self.selected_capacity.covers(self.request_capacity):
            raise SchemaError(
                "selected capacity must dominate every request dimension",
                path=f"{path}.selected_capacity",
            )
        if self.exact_match != (self.request_profile_id == self.selected_profile_id):
            raise SchemaError(
                "must equal profile-id equality", path=f"{path}.exact_match"
            )
        if self.exact_match and self.selected_capacity != self.request_capacity:
            raise SchemaError(
                "exact profile ids require identical capacities",
                path=f"{path}.selected_capacity",
            )
        if tuple(sorted(set(self.considered_profile_ids))) != self.considered_profile_ids:
            raise SchemaError(
                "must be unique and strictly sorted",
                path=f"{path}.considered_profile_ids",
            )
        if self.selected_profile_id not in self.considered_profile_ids:
            raise SchemaError(
                "must contain selected_profile_id",
                path=f"{path}.considered_profile_ids",
            )
        expected_id = stable_artifact_id(
            "stage3_profile_selection",
            self._semantic_key(),
            schema_version=STAGE3_PROFILE_SELECTION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )
