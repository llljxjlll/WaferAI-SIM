"""Exact static PD topology and per-layer KV handoff contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .stage3_profile import Stage3ProfileMode, Stage3StaticProfile


STAGE4_PD_PLAN_SCHEMA_VERSION = "wafer_frontend.stage4_pd_plan/v1alpha1"
STAGE4_PD_ORACLE_SCHEMA_VERSION = "wafer_frontend.stage4_pd_oracle/v1alpha2"


class Stage4PdMode(str, Enum):
    FUSED = "fused"
    SEPARATED = "separated"


class Stage4KvReshardKind(str, Enum):
    NONE = "none"
    ONE_TO_ONE = "one_to_one"
    GATHER = "gather"
    SCATTER = "scatter"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _validate_sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class KvHeadSlice:
    start: int
    count: int

    @property
    def end(self) -> int:
        return self.start + self.count

    def validate(self, path: str = "head_slice") -> None:
        validate_uint64(self.start, f"{path}.start")
        _positive(self.count, f"{path}.count")
        validate_uint64(self.end, f"{path}.derived.end")


@dataclass(frozen=True, slots=True)
class Stage4KvRankFlow:
    source_rank: int
    destination_rank: int
    head_slice: KvHeadSlice
    bytes: int

    def validate(self, path: str = "rank_flow") -> None:
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        if type(self.head_slice) is not KvHeadSlice:
            raise SchemaError(
                "must be a KvHeadSlice", path=f"{path}.head_slice"
            )
        self.head_slice.validate(f"{path}.head_slice")
        _positive(self.bytes, f"{path}.bytes")

    def validate_against(
        self,
        *,
        prefill_tp: int,
        decode_tp: int,
        num_kv_heads: int,
        expected_bytes_per_head: int,
        path: str,
    ) -> None:
        self.validate(path)
        if self.source_rank >= prefill_tp:
            raise SchemaError(
                "must be smaller than prefill_tp",
                path=f"{path}.source_rank",
            )
        if self.destination_rank >= decode_tp:
            raise SchemaError(
                "must be smaller than decode_tp",
                path=f"{path}.destination_rank",
            )
        if self.head_slice.end > num_kv_heads:
            raise SchemaError(
                "must stay inside num_kv_heads",
                path=f"{path}.head_slice",
            )
        expected_bytes = expected_bytes_per_head * self.head_slice.count
        validate_uint64(expected_bytes, f"{path}.derived.bytes")
        if self.bytes != expected_bytes:
            raise SchemaError(
                f"must equal {expected_bytes}", path=f"{path}.bytes"
            )


@dataclass(frozen=True, slots=True)
class Stage4KvLayerHandoff:
    layer_index: int
    request_ref: str
    token_count: int
    logical_unique_bytes: int
    flows: tuple[Stage4KvRankFlow, ...]

    @property
    def delivered_bytes(self) -> int:
        return sum(flow.bytes for flow in self.flows)

    @property
    def state_transfer_count(self) -> int:
        # The analytic rank flow combines K and V; executable transport uses
        # one exact state-transfer contract for each tensor kind.
        return 2 * len(self.flows)

    def validate(self, path: str = "layer_handoff") -> None:
        validate_uint64(self.layer_index, f"{path}.layer_index")
        validate_nonempty(self.request_ref, f"{path}.request_ref")
        _positive(self.token_count, f"{path}.token_count")
        _positive(self.logical_unique_bytes, f"{path}.logical_unique_bytes")
        if type(self.flows) is not tuple or not self.flows:
            raise SchemaError(
                "must be a non-empty immutable tuple", path=f"{path}.flows"
            )
        for index, flow in enumerate(self.flows):
            if type(flow) is not Stage4KvRankFlow:
                raise SchemaError(
                    "must be a Stage4KvRankFlow",
                    path=f"{path}.flows[{index}]",
                )
            flow.validate(f"{path}.flows[{index}]")
        order = tuple(
            sorted(
                self.flows,
                key=lambda item: (
                    item.source_rank,
                    item.destination_rank,
                    item.head_slice.start,
                ),
            )
        )
        if self.flows != order:
            raise SchemaError(
                "must use canonical source/destination/head order",
                path=f"{path}.flows",
            )

    def validate_against(
        self,
        *,
        num_layers: int,
        prefill_tp: int,
        decode_tp: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int,
        path: str,
    ) -> None:
        self.validate(path)
        if self.layer_index >= num_layers:
            raise SchemaError(
                "must be smaller than num_layers",
                path=f"{path}.layer_index",
            )
        expected_unique = (
            2
            * self.token_count
            * num_kv_heads
            * head_dim
            * dtype_bytes
        )
        validate_uint64(
            expected_unique, f"{path}.derived.logical_unique_bytes"
        )
        if self.logical_unique_bytes != expected_unique:
            raise SchemaError(
                f"must equal {expected_unique}",
                path=f"{path}.logical_unique_bytes",
            )
        expected_bytes_per_head = (
            2 * self.token_count * head_dim * dtype_bytes
        )
        for index, flow in enumerate(self.flows):
            flow.validate_against(
                prefill_tp=prefill_tp,
                decode_tp=decode_tp,
                num_kv_heads=num_kv_heads,
                expected_bytes_per_head=expected_bytes_per_head,
                path=f"{path}.flows[{index}]",
            )
        covered_heads = [
            head
            for flow in self.flows
            for head in range(flow.head_slice.start, flow.head_slice.end)
        ]
        if sorted(covered_heads) != list(range(num_kv_heads)):
            raise SchemaError(
                "flows must cover every KV head exactly once",
                path=f"{path}.flows",
            )
        if self.delivered_bytes != self.logical_unique_bytes:
            raise SchemaError(
                "non-replicated handoff must deliver exactly the logical unique bytes",
                path=f"{path}.flows",
            )


def _request_pairs(
    prefill: Stage3StaticProfile,
    decode: Stage3StaticProfile,
    *,
    path: str,
) -> tuple[tuple[object, object], ...]:
    prefill_by_ref = {request.request_ref: request for request in prefill.requests}
    decode_by_ref = {request.request_ref: request for request in decode.requests}
    if set(prefill_by_ref) != set(decode_by_ref):
        raise SchemaError(
            "prefill/decode exact profiles must have identical request refs",
            path=path,
        )
    result = []
    for request_ref in sorted(prefill_by_ref):
        prefill_request = prefill_by_ref[request_ref]
        decode_request = decode_by_ref[request_ref]
        if (
            decode_request.context_tokens
            != prefill_request.context_tokens + decode_request.query_tokens
        ):
            raise SchemaError(
                "decode context must immediately follow the prefill context",
                path=f"{path}.{request_ref}.context_tokens",
            )
        if (
            prefill_request.kv_span.page_start
            != decode_request.kv_span.page_start
            or prefill_request.kv_span.page_size_tokens
            != decode_request.kv_span.page_size_tokens
        ):
            raise SchemaError(
                "prefill/decode KV page lineage must preserve start and page size",
                path=f"{path}.{request_ref}.kv_span",
            )
        result.append((prefill_request, decode_request))
    return tuple(result)


def _expected_flows(
    *,
    token_count: int,
    prefill_tp: int,
    decode_tp: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> tuple[Stage4KvRankFlow, ...]:
    source_width = num_kv_heads // prefill_tp
    destination_width = num_kv_heads // decode_tp
    result: list[Stage4KvRankFlow] = []
    for source_rank in range(prefill_tp):
        source_start = source_rank * source_width
        source_end = source_start + source_width
        for destination_rank in range(decode_tp):
            destination_start = destination_rank * destination_width
            destination_end = destination_start + destination_width
            start = max(source_start, destination_start)
            end = min(source_end, destination_end)
            if start >= end:
                continue
            count = end - start
            result.append(
                Stage4KvRankFlow(
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    head_slice=KvHeadSlice(start=start, count=count),
                    bytes=2 * token_count * count * head_dim * dtype_bytes,
                )
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Stage4PdPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_spec_digest: str
    mode: Stage4PdMode
    reshard: Stage4KvReshardKind
    prefill_instance_ref: str
    decode_instance_ref: str
    prefill_tp: int
    decode_tp: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: DType
    prefill_profile: Stage3StaticProfile
    decode_profile: Stage3StaticProfile
    handoffs: tuple[Stage4KvLayerHandoff, ...]

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage4PdPlan":
        result = cls(
            schema_version=STAGE4_PD_PLAN_SCHEMA_VERSION,
            producer_pass="stage4_pd_plan",
            id=stable_artifact_id(
                "stage4_pd_plan",
                semantic_key,
                schema_version=STAGE4_PD_PLAN_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_spec_digest": self.source_spec_digest,
            "mode": self.mode,
            "reshard": self.reshard,
            "prefill_instance_ref": self.prefill_instance_ref,
            "decode_instance_ref": self.decode_instance_ref,
            "prefill_tp": self.prefill_tp,
            "decode_tp": self.decode_tp,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "prefill_profile": self.prefill_profile,
            "decode_profile": self.decode_profile,
            "handoffs": self.handoffs,
        }

    def validate(self, path: str = "stage4_pd_plan") -> None:
        if self.schema_version != STAGE4_PD_PLAN_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage4_pd_plan":
            raise SchemaError(
                "must be 'stage4_pd_plan'", path=f"{path}.producer_pass"
            )
        _validate_sha256(self.source_spec_digest, f"{path}.source_spec_digest")
        if type(self.mode) is not Stage4PdMode:
            raise SchemaError("must be a Stage4PdMode", path=f"{path}.mode")
        if type(self.reshard) is not Stage4KvReshardKind:
            raise SchemaError(
                "must be a Stage4KvReshardKind", path=f"{path}.reshard"
            )
        validate_nonempty(
            self.prefill_instance_ref, f"{path}.prefill_instance_ref"
        )
        validate_nonempty(
            self.decode_instance_ref, f"{path}.decode_instance_ref"
        )
        for field_name in (
            "prefill_tp",
            "decode_tp",
            "num_layers",
            "num_kv_heads",
            "head_dim",
        ):
            _positive(getattr(self, field_name), f"{path}.{field_name}")
        if (
            self.num_kv_heads % self.prefill_tp
            or self.num_kv_heads % self.decode_tp
        ):
            raise SchemaError(
                "num_kv_heads must divide evenly across both TP groups",
                path=path,
            )
        if self.dtype is not DType.FP16:
            raise SchemaError("Stage 4 v1 requires FP16", path=f"{path}.dtype")
        if type(self.prefill_profile) is not Stage3StaticProfile:
            raise SchemaError(
                "must be a Stage3StaticProfile",
                path=f"{path}.prefill_profile",
            )
        if type(self.decode_profile) is not Stage3StaticProfile:
            raise SchemaError(
                "must be a Stage3StaticProfile",
                path=f"{path}.decode_profile",
            )
        self.prefill_profile.validate(f"{path}.prefill_profile")
        self.decode_profile.validate(f"{path}.decode_profile")
        if self.prefill_profile.mode is not Stage3ProfileMode.PREFILL:
            raise SchemaError(
                "must be pure prefill", path=f"{path}.prefill_profile"
            )
        if self.decode_profile.mode is not Stage3ProfileMode.DECODE:
            raise SchemaError(
                "must be pure decode", path=f"{path}.decode_profile"
            )
        pairs = _request_pairs(
            self.prefill_profile,
            self.decode_profile,
            path=f"{path}.request_lineage",
        )
        if self.mode is Stage4PdMode.FUSED:
            if (
                self.prefill_instance_ref != self.decode_instance_ref
                or self.prefill_tp != self.decode_tp
                or self.reshard is not Stage4KvReshardKind.NONE
                or self.handoffs
            ):
                raise SchemaError(
                    "fused PD requires one instance, equal TP, no reshard, and no handoff",
                    path=path,
                )
        else:
            if self.prefill_instance_ref == self.decode_instance_ref:
                raise SchemaError(
                    "separated PD requires distinct instances", path=path
                )
            expected_reshard = (
                Stage4KvReshardKind.ONE_TO_ONE
                if self.prefill_tp == self.decode_tp
                else (
                    Stage4KvReshardKind.GATHER
                    if self.prefill_tp > self.decode_tp
                    else Stage4KvReshardKind.SCATTER
                )
            )
            if self.reshard is not expected_reshard:
                raise SchemaError(
                    f"must equal {expected_reshard.value!r}",
                    path=f"{path}.reshard",
                )
            expected_handoffs = tuple(
                Stage4KvLayerHandoff(
                    layer_index=layer_index,
                    request_ref=prefill_request.request_ref,
                    token_count=prefill_request.context_tokens,
                    logical_unique_bytes=(
                        2
                        * prefill_request.context_tokens
                        * self.num_kv_heads
                        * self.head_dim
                        * 2
                    ),
                    flows=_expected_flows(
                        token_count=prefill_request.context_tokens,
                        prefill_tp=self.prefill_tp,
                        decode_tp=self.decode_tp,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        dtype_bytes=2,
                    ),
                )
                for layer_index in range(self.num_layers)
                for prefill_request, _decode_request in pairs
            )
            if self.handoffs != expected_handoffs:
                raise SchemaError(
                    "must exactly equal the per-layer request/head intersection matrix",
                    path=f"{path}.handoffs",
                )
            for index, handoff in enumerate(self.handoffs):
                handoff.validate_against(
                    num_layers=self.num_layers,
                    prefill_tp=self.prefill_tp,
                    decode_tp=self.decode_tp,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    dtype_bytes=2,
                    path=f"{path}.handoffs[{index}]",
                )
        expected_id = stable_artifact_id(
            "stage4_pd_plan",
            self._semantic_key(),
            schema_version=STAGE4_PD_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class Stage4KvHandoffMetrics:
    layer_index: int
    request_ref: str
    logical_unique_bytes: int
    delivered_bytes: int
    rank_flow_count: int
    state_transfer_count: int

    def validate(self, path: str) -> None:
        validate_uint64(self.layer_index, f"{path}.layer_index")
        validate_nonempty(self.request_ref, f"{path}.request_ref")
        for field_name in (
            "logical_unique_bytes",
            "delivered_bytes",
            "rank_flow_count",
            "state_transfer_count",
        ):
            _positive(getattr(self, field_name), f"{path}.{field_name}")
        if self.delivered_bytes != self.logical_unique_bytes:
            raise SchemaError(
                "v1 has no KV-head replication", path=f"{path}.delivered_bytes"
            )
        if self.state_transfer_count != 2 * self.rank_flow_count:
            raise SchemaError(
                "must equal two K/V transfers per rank flow",
                path=f"{path}.state_transfer_count",
            )


@dataclass(frozen=True, slots=True)
class Stage4KvEndpointPairMetrics:
    source_rank: int
    destination_rank: int
    logical_unique_bytes: int
    delivered_bytes: int
    state_transfer_count: int

    def validate(self, path: str) -> None:
        validate_uint64(self.source_rank, f"{path}.source_rank")
        validate_uint64(self.destination_rank, f"{path}.destination_rank")
        for field_name in (
            "logical_unique_bytes",
            "delivered_bytes",
            "state_transfer_count",
        ):
            _positive(getattr(self, field_name), f"{path}.{field_name}")
        if self.delivered_bytes != self.logical_unique_bytes:
            raise SchemaError(
                "v1 has no KV-head replication", path=f"{path}.delivered_bytes"
            )
        if self.state_transfer_count % 2:
            raise SchemaError(
                "must contain complete K/V transfer pairs",
                path=f"{path}.state_transfer_count",
            )


def _expected_endpoint_pair_metrics(
    plan: Stage4PdPlan,
) -> tuple[Stage4KvEndpointPairMetrics, ...]:
    aggregates: dict[tuple[int, int], list[int]] = {}
    for handoff in plan.handoffs:
        for flow in handoff.flows:
            key = (flow.source_rank, flow.destination_rank)
            values = aggregates.setdefault(key, [0, 0, 0])
            values[0] += flow.bytes
            values[1] += flow.bytes
            values[2] += 2
    return tuple(
        Stage4KvEndpointPairMetrics(
            source_rank=source_rank,
            destination_rank=destination_rank,
            logical_unique_bytes=values[0],
            delivered_bytes=values[1],
            state_transfer_count=values[2],
        )
        for (source_rank, destination_rank), values in sorted(aggregates.items())
    )


@dataclass(frozen=True, slots=True)
class Stage4PdOracle:
    schema_version: str
    producer_pass: str
    id: str
    source_plan_id: str
    mode: Stage4PdMode
    reshard: Stage4KvReshardKind
    handoff_applicable: bool
    metrics: tuple[Stage4KvHandoffMetrics, ...]
    endpoint_pair_metrics: tuple[Stage4KvEndpointPairMetrics, ...]
    unique_endpoint_route_count: int
    logical_unique_bytes: int
    delivered_bytes: int
    rank_flow_count: int
    state_transfer_count: int
    decode_wait_dependency_count: int

    @classmethod
    def create(cls, **semantic_key: object) -> "Stage4PdOracle":
        result = cls(
            schema_version=STAGE4_PD_ORACLE_SCHEMA_VERSION,
            producer_pass="stage4_pd_oracle",
            id=stable_artifact_id(
                "stage4_pd_oracle",
                semantic_key,
                schema_version=STAGE4_PD_ORACLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_plan_id": self.source_plan_id,
            "mode": self.mode,
            "reshard": self.reshard,
            "handoff_applicable": self.handoff_applicable,
            "metrics": self.metrics,
            "endpoint_pair_metrics": self.endpoint_pair_metrics,
            "unique_endpoint_route_count": self.unique_endpoint_route_count,
            "logical_unique_bytes": self.logical_unique_bytes,
            "delivered_bytes": self.delivered_bytes,
            "rank_flow_count": self.rank_flow_count,
            "state_transfer_count": self.state_transfer_count,
            "decode_wait_dependency_count": self.decode_wait_dependency_count,
        }

    def validate(self, path: str = "stage4_pd_oracle") -> None:
        if self.schema_version != STAGE4_PD_ORACLE_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "stage4_pd_oracle":
            raise SchemaError(
                "must be 'stage4_pd_oracle'", path=f"{path}.producer_pass"
            )
        validate_nonempty(self.source_plan_id, f"{path}.source_plan_id")
        if type(self.mode) is not Stage4PdMode:
            raise SchemaError("must be a Stage4PdMode", path=f"{path}.mode")
        if type(self.reshard) is not Stage4KvReshardKind:
            raise SchemaError(
                "must be a Stage4KvReshardKind", path=f"{path}.reshard"
            )
        if type(self.handoff_applicable) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.handoff_applicable")
        if type(self.metrics) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.metrics")
        for index, metric in enumerate(self.metrics):
            if type(metric) is not Stage4KvHandoffMetrics:
                raise SchemaError(
                    "must be a Stage4KvHandoffMetrics",
                    path=f"{path}.metrics[{index}]",
                )
            metric.validate(f"{path}.metrics[{index}]")
        if type(self.endpoint_pair_metrics) is not tuple:
            raise SchemaError(
                "must be an immutable tuple",
                path=f"{path}.endpoint_pair_metrics",
            )
        for index, metric in enumerate(self.endpoint_pair_metrics):
            if type(metric) is not Stage4KvEndpointPairMetrics:
                raise SchemaError(
                    "must be a Stage4KvEndpointPairMetrics",
                    path=f"{path}.endpoint_pair_metrics[{index}]",
                )
            metric.validate(f"{path}.endpoint_pair_metrics[{index}]")
        endpoint_pair_order = tuple(
            sorted(
                self.endpoint_pair_metrics,
                key=lambda item: (item.source_rank, item.destination_rank),
            )
        )
        if self.endpoint_pair_metrics != endpoint_pair_order:
            raise SchemaError(
                "must use canonical source/destination order",
                path=f"{path}.endpoint_pair_metrics",
            )
        endpoint_pairs = tuple(
            (metric.source_rank, metric.destination_rank)
            for metric in self.endpoint_pair_metrics
        )
        if len(set(endpoint_pairs)) != len(endpoint_pairs):
            raise SchemaError(
                "must contain each endpoint pair exactly once",
                path=f"{path}.endpoint_pair_metrics",
            )
        totals = (
            sum(metric.logical_unique_bytes for metric in self.metrics),
            sum(metric.delivered_bytes for metric in self.metrics),
            sum(metric.rank_flow_count for metric in self.metrics),
            sum(metric.state_transfer_count for metric in self.metrics),
        )
        observed = (
            self.logical_unique_bytes,
            self.delivered_bytes,
            self.rank_flow_count,
            self.state_transfer_count,
        )
        for field_name, value in zip(
            (
                "unique_endpoint_route_count",
                "logical_unique_bytes",
                "delivered_bytes",
                "rank_flow_count",
                "state_transfer_count",
                "decode_wait_dependency_count",
            ),
            (self.unique_endpoint_route_count,)
            + observed
            + (self.decode_wait_dependency_count,),
        ):
            validate_uint64(value, f"{path}.{field_name}")
        if observed != totals:
            raise SchemaError(
                "aggregate metrics must exactly sum per-handoff metrics",
                path=path,
            )
        if self.unique_endpoint_route_count != len(self.endpoint_pair_metrics):
            raise SchemaError(
                "must equal the number of unique endpoint pairs",
                path=f"{path}.unique_endpoint_route_count",
            )
        endpoint_totals = (
            sum(
                metric.logical_unique_bytes
                for metric in self.endpoint_pair_metrics
            ),
            sum(metric.delivered_bytes for metric in self.endpoint_pair_metrics),
            sum(
                metric.state_transfer_count
                for metric in self.endpoint_pair_metrics
            ),
        )
        if endpoint_totals != (
            self.logical_unique_bytes,
            self.delivered_bytes,
            self.state_transfer_count,
        ):
            raise SchemaError(
                "endpoint-pair aggregates must close oracle totals",
                path=f"{path}.endpoint_pair_metrics",
            )
        if self.decode_wait_dependency_count != self.state_transfer_count:
            raise SchemaError(
                "decode must wait for every K/V state transfer",
                path=f"{path}.decode_wait_dependency_count",
            )
        if self.mode is Stage4PdMode.FUSED:
            if (
                self.reshard is not Stage4KvReshardKind.NONE
                or self.handoff_applicable
                or self.metrics
                or self.endpoint_pair_metrics
                or self.unique_endpoint_route_count
                or any(observed)
                or self.decode_wait_dependency_count
            ):
                raise SchemaError(
                    "fused PD must report handoff as N/A with zero traffic",
                    path=path,
                )
        elif (
            not self.handoff_applicable
            or not self.metrics
            or not self.endpoint_pair_metrics
            or not self.unique_endpoint_route_count
        ):
            raise SchemaError(
                "separated PD requires applicable non-empty handoff metrics",
                path=path,
            )
        expected_id = stable_artifact_id(
            "stage4_pd_oracle",
            self._semantic_key(),
            schema_version=STAGE4_PD_ORACLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self, plan: Stage4PdPlan, path: str = "stage4_pd_oracle"
    ) -> None:
        self.validate(path)
        plan.validate("stage4_pd_plan")
        expected_metrics = tuple(
            Stage4KvHandoffMetrics(
                layer_index=handoff.layer_index,
                request_ref=handoff.request_ref,
                logical_unique_bytes=handoff.logical_unique_bytes,
                delivered_bytes=handoff.delivered_bytes,
                rank_flow_count=len(handoff.flows),
                state_transfer_count=handoff.state_transfer_count,
            )
            for handoff in plan.handoffs
        )
        expected_endpoint_pair_metrics = _expected_endpoint_pair_metrics(plan)
        expected = {
            "source_plan_id": plan.id,
            "mode": plan.mode,
            "reshard": plan.reshard,
            "handoff_applicable": plan.mode is Stage4PdMode.SEPARATED,
            "metrics": expected_metrics,
            "endpoint_pair_metrics": expected_endpoint_pair_metrics,
            "unique_endpoint_route_count": len(expected_endpoint_pair_metrics),
            "logical_unique_bytes": sum(
                metric.logical_unique_bytes for metric in expected_metrics
            ),
            "delivered_bytes": sum(
                metric.delivered_bytes for metric in expected_metrics
            ),
            "rank_flow_count": sum(
                metric.rank_flow_count for metric in expected_metrics
            ),
            "state_transfer_count": sum(
                metric.state_transfer_count for metric in expected_metrics
            ),
            "decode_wait_dependency_count": sum(
                metric.state_transfer_count for metric in expected_metrics
            ),
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise SchemaError(
                    "differs from the source plan",
                    path=f"{path}.{field_name}",
                )


__all__ = [
    "STAGE4_PD_PLAN_SCHEMA_VERSION",
    "STAGE4_PD_ORACLE_SCHEMA_VERSION",
    "KvHeadSlice",
    "Stage4KvEndpointPairMetrics",
    "Stage4KvHandoffMetrics",
    "Stage4KvLayerHandoff",
    "Stage4KvRankFlow",
    "Stage4KvReshardKind",
    "Stage4PdMode",
    "Stage4PdOracle",
    "Stage4PdPlan",
]
