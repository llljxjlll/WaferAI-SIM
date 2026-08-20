"""Build exact Stage 4 PD topology and analytic KV handoff evidence."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.experiment import ExperimentSpec, InferSource
from ..schema.serde import canonical_digest
from ..schema.stage3_profile import Stage3StaticProfile
from ..schema.stage4_pd import (
    KvHeadSlice,
    Stage4KvEndpointPairMetrics,
    Stage4KvHandoffMetrics,
    Stage4KvLayerHandoff,
    Stage4KvRankFlow,
    Stage4KvReshardKind,
    Stage4PdMode,
    Stage4PdOracle,
    Stage4PdPlan,
)


def _flows(
    *,
    token_count: int,
    prefill_tp: int,
    decode_tp: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[Stage4KvRankFlow, ...]:
    source_heads = num_kv_heads // prefill_tp
    destination_heads = num_kv_heads // decode_tp
    flows: list[Stage4KvRankFlow] = []
    for source_rank in range(prefill_tp):
        source_start = source_rank * source_heads
        source_stop = source_start + source_heads
        for destination_rank in range(decode_tp):
            destination_start = destination_rank * destination_heads
            destination_stop = destination_start + destination_heads
            start = max(source_start, destination_start)
            stop = min(source_stop, destination_stop)
            if start >= stop:
                continue
            head_count = stop - start
            flows.append(
                Stage4KvRankFlow(
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    head_slice=KvHeadSlice(start=start, count=head_count),
                    bytes=2 * token_count * head_count * head_dim * 2,
                )
            )
    return tuple(flows)


def build_stage4_pd_plan(
    spec: ExperimentSpec,
    *,
    prefill_profile: Stage3StaticProfile,
    decode_profile: Stage3StaticProfile,
) -> Stage4PdPlan:
    """Derive one selected fused/separated PD plan from exact profiles."""

    if type(spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="spec")
    spec.validate("spec")
    infer = spec.workload.infer
    if infer.source is not InferSource.PD_STATIC or infer.pd_static is None:
        raise SchemaError(
            "Stage 4 PD planning requires infer.source='pd_static'",
            path="spec.workload.infer.source",
        )
    if type(prefill_profile) is not Stage3StaticProfile:
        raise SchemaError(
            "must be a Stage3StaticProfile", path="prefill_profile"
        )
    if type(decode_profile) is not Stage3StaticProfile:
        raise SchemaError(
            "must be a Stage3StaticProfile", path="decode_profile"
        )
    prefill_profile.validate("prefill_profile")
    decode_profile.validate("decode_profile")
    pd_static = infer.pd_static
    if prefill_profile.key != pd_static.prefill_profile:
        raise SchemaError(
            "key differs from spec.workload.infer.pd_static.prefill_profile",
            path="prefill_profile.key",
        )
    if decode_profile.key != pd_static.decode_profile:
        raise SchemaError(
            "key differs from spec.workload.infer.pd_static.decode_profile",
            path="decode_profile.key",
        )
    instances = {instance.id: instance for instance in spec.parallel.instances}
    prefill_instance = instances[pd_static.prefill_instance_ref]
    decode_instance = instances[pd_static.decode_instance_ref]
    fused = prefill_instance.id == decode_instance.id
    mode = Stage4PdMode.FUSED if fused else Stage4PdMode.SEPARATED
    reshard = (
        Stage4KvReshardKind.NONE
        if fused
        else (
            Stage4KvReshardKind.ONE_TO_ONE
            if prefill_instance.tp == decode_instance.tp
            else (
                Stage4KvReshardKind.GATHER
                if prefill_instance.tp > decode_instance.tp
                else Stage4KvReshardKind.SCATTER
            )
        )
    )
    prefill_requests = {
        request.request_ref: request for request in prefill_profile.requests
    }
    decode_requests = {
        request.request_ref: request for request in decode_profile.requests
    }
    if set(prefill_requests) != set(decode_requests):
        raise SchemaError(
            "prefill/decode exact profiles must have identical request refs",
            path="decode_profile.requests",
        )
    handoffs = ()
    if not fused:
        handoffs = tuple(
            Stage4KvLayerHandoff(
                layer_index=layer_index,
                request_ref=request_ref,
                token_count=prefill_requests[request_ref].context_tokens,
                logical_unique_bytes=(
                    2
                    * prefill_requests[request_ref].context_tokens
                    * spec.model.KVH
                    * spec.model.DH
                    * 2
                ),
                flows=_flows(
                    token_count=prefill_requests[request_ref].context_tokens,
                    prefill_tp=prefill_instance.tp,
                    decode_tp=decode_instance.tp,
                    num_kv_heads=spec.model.KVH,
                    head_dim=spec.model.DH,
                ),
            )
            for layer_index in range(spec.model.L)
            for request_ref in sorted(prefill_requests)
        )
    plan = Stage4PdPlan.create(
        source_spec_digest=canonical_digest(spec),
        mode=mode,
        reshard=reshard,
        prefill_instance_ref=prefill_instance.id,
        decode_instance_ref=decode_instance.id,
        prefill_tp=prefill_instance.tp,
        decode_tp=decode_instance.tp,
        num_layers=spec.model.L,
        num_kv_heads=spec.model.KVH,
        head_dim=spec.model.DH,
        dtype=spec.model.dtype,
        prefill_profile=prefill_profile,
        decode_profile=decode_profile,
        handoffs=handoffs,
    )
    plan.validate()
    return plan


def build_stage4_pd_oracle(plan: Stage4PdPlan) -> Stage4PdOracle:
    """Build aggregate evidence without trusting caller-supplied totals."""

    if type(plan) is not Stage4PdPlan:
        raise SchemaError("must be a Stage4PdPlan", path="plan")
    plan.validate("plan")
    metrics = tuple(
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
    endpoint_aggregates: dict[tuple[int, int], list[int]] = {}
    for handoff in plan.handoffs:
        for flow in handoff.flows:
            key = (flow.source_rank, flow.destination_rank)
            aggregate = endpoint_aggregates.setdefault(key, [0, 0, 0])
            aggregate[0] += flow.bytes
            aggregate[1] += flow.bytes
            aggregate[2] += 2
    endpoint_pair_metrics = tuple(
        Stage4KvEndpointPairMetrics(
            source_rank=source_rank,
            destination_rank=destination_rank,
            logical_unique_bytes=aggregate[0],
            delivered_bytes=aggregate[1],
            state_transfer_count=aggregate[2],
        )
        for (source_rank, destination_rank), aggregate in sorted(
            endpoint_aggregates.items()
        )
    )
    oracle = Stage4PdOracle.create(
        source_plan_id=plan.id,
        mode=plan.mode,
        reshard=plan.reshard,
        handoff_applicable=plan.mode is Stage4PdMode.SEPARATED,
        metrics=metrics,
        endpoint_pair_metrics=endpoint_pair_metrics,
        unique_endpoint_route_count=len(endpoint_pair_metrics),
        logical_unique_bytes=sum(
            metric.logical_unique_bytes for metric in metrics
        ),
        delivered_bytes=sum(metric.delivered_bytes for metric in metrics),
        rank_flow_count=sum(metric.rank_flow_count for metric in metrics),
        state_transfer_count=sum(
            metric.state_transfer_count for metric in metrics
        ),
        decode_wait_dependency_count=sum(
            metric.state_transfer_count for metric in metrics
        ),
    )
    oracle.validate_against(plan)
    return oracle


__all__ = ["build_stage4_pd_oracle", "build_stage4_pd_plan"]
