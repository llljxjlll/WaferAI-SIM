"""Isolated typed contract for the S3-Lite static-route MoE case."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_uint64
from .serde import canonical_digest


S3_LITE_BASELINE_EPOCH = "s3-lite-v1"
S3_LITE_STATIC_MOE_CASE_ID = "case.s3_lite.static_moe_infer"
LITE_MOE_TRACE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_trace/v1alpha1"
)
LITE_MOE_SPEC_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_spec/v1alpha1"
)
LITE_MOE_ORACLE_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_static_moe_oracle/v1alpha1"
)

_DIE_COUNT = 2
_EP_DEGREE = 2
_EXPERT_COUNT = 4
_TOP_K = 1
_DTYPE_BYTES = {DType.FP16: 2}


class LiteMoeRoutingKind(str, Enum):
    STATIC_TRACE = "static_trace"
    RANDOM = "random"
    GATE_TOPK = "gate_topk"


class LiteMoeTransferRole(str, Enum):
    MOE_DISPATCH = "moe_dispatch"
    MOE_COMBINE = "moe_combine"


_ROLE_RANK = {
    LiteMoeTransferRole.MOE_DISPATCH: 0,
    LiteMoeTransferRole.MOE_COMBINE: 1,
}


@dataclass(frozen=True, slots=True)
class LiteMoeTraceAssignment:
    token_index: int
    expert_index: int
    slot_index: int

    def validate(self, path: str = "lite_moe_trace_assignment") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.expert_index >= _EXPERT_COUNT:
            raise SchemaError(
                f"must be less than {_EXPERT_COUNT}",
                path=f"{path}.expert_index",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeStaticTrace:
    schema_version: str
    producer_pass: str
    id: str
    token_count: int
    assignments: tuple[LiteMoeTraceAssignment, ...]
    expert_histogram: tuple[int, int, int, int]

    @classmethod
    def create(
        cls,
        *,
        token_count: int,
        assignments: tuple[LiteMoeTraceAssignment, ...],
        expert_histogram: tuple[int, int, int, int],
    ) -> "LiteMoeStaticTrace":
        semantic_key = {
            "token_count": token_count,
            "assignments": assignments,
            "expert_histogram": expert_histogram,
        }
        result = cls(
            schema_version=LITE_MOE_TRACE_SCHEMA_VERSION,
            producer_pass="lite_moe_static_trace",
            id=stable_artifact_id(
                "s3_lite_static_moe_trace",
                semantic_key,
                schema_version=LITE_MOE_TRACE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "assignments": self.assignments,
            "expert_histogram": self.expert_histogram,
        }

    def validate(self, path: str = "lite_moe_static_trace") -> None:
        if self.schema_version != LITE_MOE_TRACE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_static_trace":
            raise SchemaError(
                "must be 'lite_moe_static_trace'",
                path=f"{path}.producer_pass",
            )
        validate_uint64(self.token_count, f"{path}.token_count")
        if self.token_count == 0:
            raise SchemaError("must be positive", path=f"{path}.token_count")
        if type(self.assignments) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.assignments"
            )
        if len(self.assignments) != self.token_count:
            raise SchemaError(
                "must contain exactly one assignment per token",
                path=f"{path}.assignments",
            )
        token_indices: list[int] = []
        slots_by_expert: list[list[int]] = [[], [], [], []]
        for index, assignment in enumerate(self.assignments):
            assignment_path = f"{path}.assignments[{index}]"
            if type(assignment) is not LiteMoeTraceAssignment:
                raise SchemaError(
                    "must be a LiteMoeTraceAssignment", path=assignment_path
                )
            assignment.validate(assignment_path)
            token_indices.append(assignment.token_index)
            slots_by_expert[assignment.expert_index].append(
                assignment.slot_index
            )
        if tuple(token_indices) != tuple(range(self.token_count)):
            raise SchemaError(
                "must cover every token exactly once in token order",
                path=f"{path}.assignments",
            )
        for expert_index, slots in enumerate(slots_by_expert):
            if tuple(slots) != tuple(range(len(slots))):
                raise SchemaError(
                    "expert slots must be unique, gap-free, and canonical",
                    path=f"{path}.assignments",
                )
        if (
            type(self.expert_histogram) is not tuple
            or len(self.expert_histogram) != _EXPERT_COUNT
        ):
            raise SchemaError(
                "must contain exactly four expert counts",
                path=f"{path}.expert_histogram",
            )
        for index, count in enumerate(self.expert_histogram):
            validate_uint64(count, f"{path}.expert_histogram[{index}]")
        observed_histogram = tuple(len(slots) for slots in slots_by_expert)
        if self.expert_histogram != observed_histogram:
            raise SchemaError(
                "must equal the assignment histogram",
                path=f"{path}.expert_histogram",
            )
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_trace",
            self._semantic_key(),
            schema_version=LITE_MOE_TRACE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeSpec:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    die_count: int
    ep_degree: int
    expert_count: int
    top_k: int
    routing_kind: LiteMoeRoutingKind
    allow_overflow: bool
    drop_tokens: bool
    hidden_size: int
    intermediate_size: int
    dtype: DType
    capacity_per_expert: int
    trace: LiteMoeStaticTrace

    @classmethod
    def create(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        capacity_per_expert: int,
        trace: LiteMoeStaticTrace,
        case_id: str = S3_LITE_STATIC_MOE_CASE_ID,
        die_count: int = _DIE_COUNT,
        ep_degree: int = _EP_DEGREE,
        expert_count: int = _EXPERT_COUNT,
        top_k: int = _TOP_K,
        routing_kind: LiteMoeRoutingKind = LiteMoeRoutingKind.STATIC_TRACE,
        allow_overflow: bool = False,
        drop_tokens: bool = False,
        dtype: DType = DType.FP16,
    ) -> "LiteMoeSpec":
        semantic_key = {
            "case_id": case_id,
            "die_count": die_count,
            "ep_degree": ep_degree,
            "expert_count": expert_count,
            "top_k": top_k,
            "routing_kind": routing_kind,
            "allow_overflow": allow_overflow,
            "drop_tokens": drop_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "dtype": dtype,
            "capacity_per_expert": capacity_per_expert,
            "trace": trace,
        }
        result = cls(
            schema_version=LITE_MOE_SPEC_SCHEMA_VERSION,
            producer_pass="lite_moe_spec",
            id=stable_artifact_id(
                "s3_lite_static_moe_spec",
                semantic_key,
                schema_version=LITE_MOE_SPEC_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_spec") -> None:
        if self.schema_version != LITE_MOE_SPEC_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_spec":
            raise SchemaError(
                "must be 'lite_moe_spec'", path=f"{path}.producer_pass"
            )
        if self.case_id != S3_LITE_STATIC_MOE_CASE_ID:
            raise SchemaError(
                f"must be {S3_LITE_STATIC_MOE_CASE_ID!r}",
                path=f"{path}.case_id",
            )
        exact_numbers = {
            "die_count": _DIE_COUNT,
            "ep_degree": _EP_DEGREE,
            "expert_count": _EXPERT_COUNT,
            "top_k": _TOP_K,
        }
        for field_name, expected in exact_numbers.items():
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value != expected:
                raise SchemaError(
                    f"S3-Lite requires exactly {expected}",
                    path=f"{path}.{field_name}",
                )
        if type(self.routing_kind) is not LiteMoeRoutingKind:
            raise SchemaError(
                "must be a LiteMoeRoutingKind", path=f"{path}.routing_kind"
            )
        if self.routing_kind is not LiteMoeRoutingKind.STATIC_TRACE:
            raise SchemaError(
                "S3-Lite supports only STATIC_TRACE; RANDOM and GATE_TOPK are disabled",
                path=f"{path}.routing_kind",
            )
        for field_name in ("allow_overflow", "drop_tokens"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise SchemaError("must be a bool", path=f"{path}.{field_name}")
            if value:
                raise SchemaError(
                    "S3-Lite forbids overflow and token dropping",
                    path=f"{path}.{field_name}",
                )
        for field_name in (
            "hidden_size",
            "intermediate_size",
            "capacity_per_expert",
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0:
                raise SchemaError("must be positive", path=f"{path}.{field_name}")
        if self.dtype is not DType.FP16:
            raise SchemaError(
                "S3-Lite supports only FP16", path=f"{path}.dtype"
            )
        if type(self.trace) is not LiteMoeStaticTrace:
            raise SchemaError(
                "must be a LiteMoeStaticTrace", path=f"{path}.trace"
            )
        self.trace.validate(f"{path}.trace")
        for expert_index, count in enumerate(self.trace.expert_histogram):
            if count > self.capacity_per_expert:
                raise SchemaError(
                    "static trace exceeds expert capacity; overflow is disabled",
                    path=f"{path}.trace.expert_histogram[{expert_index}]",
                )
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_spec",
            self._semantic_key(),
            schema_version=LITE_MOE_SPEC_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeExpertMetric:
    expert_index: int
    home_die_id: int
    token_count: int
    gemm_flops: int

    def validate(self, path: str = "lite_moe_expert_metric") -> None:
        for field_name in self.__dataclass_fields__:
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.expert_index >= _EXPERT_COUNT:
            raise SchemaError("invalid expert", path=f"{path}.expert_index")
        if self.home_die_id != self.expert_index // 2:
            raise SchemaError(
                "must use canonical two-experts-per-die placement",
                path=f"{path}.home_die_id",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeP2PMetric:
    role: LiteMoeTransferRole
    source_die_id: int
    destination_die_id: int
    token_count: int
    logical_bytes: int
    hop_count: int
    byte_hop_bytes: int

    def validate(self, path: str = "lite_moe_p2p_metric") -> None:
        if type(self.role) is not LiteMoeTransferRole:
            raise SchemaError(
                "must be a LiteMoeTransferRole", path=f"{path}.role"
            )
        for field_name in self.__dataclass_fields__:
            if field_name != "role":
                validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if (
            self.source_die_id >= _DIE_COUNT
            or self.destination_die_id >= _DIE_COUNT
            or self.source_die_id == self.destination_die_id
        ):
            raise SchemaError(
                "P2P endpoints must be distinct S3-Lite dies", path=path
            )
        if self.token_count == 0 or self.logical_bytes == 0:
            raise SchemaError("P2P metric must be non-empty", path=path)
        if self.hop_count != 1:
            raise SchemaError(
                "fixed two-die route must be exactly one hop",
                path=f"{path}.hop_count",
            )
        if self.byte_hop_bytes != self.logical_bytes:
            raise SchemaError(
                "one-hop byte count must equal logical bytes",
                path=f"{path}.byte_hop_bytes",
            )


@dataclass(frozen=True, slots=True)
class LiteMoeOracle:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    source_spec_id: str
    source_spec_digest: str
    expert_metrics: tuple[LiteMoeExpertMetric, ...]
    p2p_metrics: tuple[LiteMoeP2PMetric, ...]
    total_expert_gemm_flops: int
    logical_p2p_bytes: int
    per_hop_p2p_bytes: int

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteMoeOracle":
        result = cls(
            schema_version=LITE_MOE_ORACLE_SCHEMA_VERSION,
            producer_pass="lite_moe_oracle",
            id=stable_artifact_id(
                "s3_lite_static_moe_oracle",
                semantic_key,
                schema_version=LITE_MOE_ORACLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_moe_oracle") -> None:
        if self.schema_version != LITE_MOE_ORACLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "lite_moe_oracle":
            raise SchemaError(
                "must be 'lite_moe_oracle'", path=f"{path}.producer_pass"
            )
        if self.case_id != S3_LITE_STATIC_MOE_CASE_ID:
            raise SchemaError(
                f"must be {S3_LITE_STATIC_MOE_CASE_ID!r}",
                path=f"{path}.case_id",
            )
        if not self.source_spec_id:
            raise SchemaError(
                "must identify the source spec", path=f"{path}.source_spec_id"
            )
        if (
            type(self.source_spec_digest) is not str
            or len(self.source_spec_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.source_spec_digest)
        ):
            raise SchemaError(
                "must be a canonical SHA-256 digest",
                path=f"{path}.source_spec_digest",
            )
        if type(self.expert_metrics) is not tuple or len(self.expert_metrics) != 4:
            raise SchemaError(
                "must cover exactly four experts", path=f"{path}.expert_metrics"
            )
        for index, metric in enumerate(self.expert_metrics):
            metric_path = f"{path}.expert_metrics[{index}]"
            if type(metric) is not LiteMoeExpertMetric:
                raise SchemaError(
                    "must be a LiteMoeExpertMetric", path=metric_path
                )
            metric.validate(metric_path)
            if metric.expert_index != index:
                raise SchemaError(
                    "must be in expert order", path=f"{metric_path}.expert_index"
                )
        if type(self.p2p_metrics) is not tuple:
            raise SchemaError(
                "must be an immutable tuple", path=f"{path}.p2p_metrics"
            )
        keys = []
        for index, metric in enumerate(self.p2p_metrics):
            metric_path = f"{path}.p2p_metrics[{index}]"
            if type(metric) is not LiteMoeP2PMetric:
                raise SchemaError("must be a LiteMoeP2PMetric", path=metric_path)
            metric.validate(metric_path)
            keys.append(
                (
                    _ROLE_RANK[metric.role],
                    metric.source_die_id,
                    metric.destination_die_id,
                )
            )
        if tuple(keys) != tuple(sorted(set(keys))):
            raise SchemaError(
                "must be unique and in role/endpoint order",
                path=f"{path}.p2p_metrics",
            )
        for field_name in (
            "total_expert_gemm_flops",
            "logical_p2p_bytes",
            "per_hop_p2p_bytes",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.total_expert_gemm_flops != sum(
            item.gemm_flops for item in self.expert_metrics
        ):
            raise SchemaError(
                "must equal the expert FLOP sum",
                path=f"{path}.total_expert_gemm_flops",
            )
        if self.logical_p2p_bytes != sum(
            item.logical_bytes for item in self.p2p_metrics
        ):
            raise SchemaError(
                "must equal the logical P2P sum",
                path=f"{path}.logical_p2p_bytes",
            )
        if self.per_hop_p2p_bytes != sum(
            item.byte_hop_bytes for item in self.p2p_metrics
        ):
            raise SchemaError(
                "must equal the byte-hop P2P sum",
                path=f"{path}.per_hop_p2p_bytes",
            )
        expected_id = stable_artifact_id(
            "s3_lite_static_moe_oracle",
            self._semantic_key(),
            schema_version=LITE_MOE_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self, spec: LiteMoeSpec, path: str = "lite_moe_oracle"
    ) -> None:
        self.validate(path)
        if type(spec) is not LiteMoeSpec:
            raise SchemaError("must be a LiteMoeSpec", path="lite_moe_spec")
        spec.validate("lite_moe_spec")
        if (
            self.source_spec_id != spec.id
            or self.source_spec_digest != canonical_digest(spec)
            or self.case_id != spec.case_id
        ):
            raise SchemaError(
                "does not identify the supplied spec", path=path
            )

        histogram = [0, 0, 0, 0]
        remote_counts: dict[tuple[int, int], int] = {}
        for assignment in spec.trace.assignments:
            histogram[assignment.expert_index] += 1
            source_die = assignment.token_index % _DIE_COUNT
            destination_die = assignment.expert_index // 2
            if source_die != destination_die:
                key = (source_die, destination_die)
                remote_counts[key] = remote_counts.get(key, 0) + 1
        expected_experts = tuple(
            LiteMoeExpertMetric(
                expert_index=expert_index,
                home_die_id=expert_index // 2,
                token_count=count,
                gemm_flops=(
                    6 * count * spec.hidden_size * spec.intermediate_size
                ),
            )
            for expert_index, count in enumerate(histogram)
        )
        payload_bytes = spec.hidden_size * _DTYPE_BYTES[spec.dtype]
        expected_rows = []
        for (source, destination), count in remote_counts.items():
            expected_rows.extend(
                (
                    (
                        LiteMoeTransferRole.MOE_DISPATCH,
                        source,
                        destination,
                        count,
                    ),
                    (
                        LiteMoeTransferRole.MOE_COMBINE,
                        destination,
                        source,
                        count,
                    ),
                )
            )
        expected_p2p = []
        for role, source, destination, count in sorted(
            expected_rows,
            key=lambda row: (_ROLE_RANK[row[0]], row[1], row[2]),
        ):
            logical_bytes = count * payload_bytes
            expected_p2p.append(
                LiteMoeP2PMetric(
                    role=role,
                    source_die_id=source,
                    destination_die_id=destination,
                    token_count=count,
                    logical_bytes=logical_bytes,
                    hop_count=1,
                    byte_hop_bytes=logical_bytes,
                )
            )
        expected_p2p_tuple = tuple(expected_p2p)
        if (
            self.expert_metrics != expected_experts
            or self.p2p_metrics != expected_p2p_tuple
            or self.total_expert_gemm_flops
            != sum(item.gemm_flops for item in expected_experts)
            or self.logical_p2p_bytes
            != sum(item.logical_bytes for item in expected_p2p_tuple)
            or self.per_hop_p2p_bytes
            != sum(item.byte_hop_bytes for item in expected_p2p_tuple)
        ):
            raise SchemaError(
                "does not match independently recomputed static-route work",
                path=path,
            )


__all__ = [
    "LITE_MOE_ORACLE_SCHEMA_VERSION",
    "LITE_MOE_SPEC_SCHEMA_VERSION",
    "LITE_MOE_TRACE_SCHEMA_VERSION",
    "S3_LITE_BASELINE_EPOCH",
    "S3_LITE_STATIC_MOE_CASE_ID",
    "LiteMoeExpertMetric",
    "LiteMoeOracle",
    "LiteMoeP2PMetric",
    "LiteMoeRoutingKind",
    "LiteMoeSpec",
    "LiteMoeStaticTrace",
    "LiteMoeTraceAssignment",
    "LiteMoeTransferRole",
]
