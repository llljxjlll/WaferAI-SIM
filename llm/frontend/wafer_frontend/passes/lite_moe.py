"""Independent analytic oracle for the isolated S3-Lite MoE contract."""

from __future__ import annotations

from ..schema.lite_moe import (
    LiteMoeExpertMetric,
    LiteMoeOracle,
    LiteMoeP2PMetric,
    LiteMoeSpec,
    LiteMoeTransferRole,
)
from ..schema.serde import canonical_digest


def build_lite_moe_oracle(spec: LiteMoeSpec) -> LiteMoeOracle:
    """Recompute expert work and remote traffic only from the static trace."""

    if type(spec) is not LiteMoeSpec:
        raise TypeError("spec must be a LiteMoeSpec")
    spec.validate("lite_moe_spec")

    expert_token_counts = [0, 0, 0, 0]
    dispatch_counts: dict[tuple[int, int], int] = {}
    for assignment in spec.trace.assignments:
        expert_token_counts[assignment.expert_index] += 1
        token_home_die = assignment.token_index % 2
        expert_home_die = assignment.expert_index // 2
        if token_home_die != expert_home_die:
            endpoint = (token_home_die, expert_home_die)
            dispatch_counts[endpoint] = dispatch_counts.get(endpoint, 0) + 1

    expert_metrics = tuple(
        LiteMoeExpertMetric(
            expert_index=expert_index,
            home_die_id=expert_index // 2,
            token_count=token_count,
            gemm_flops=(
                token_count
                * spec.hidden_size
                * spec.intermediate_size
                * 6
            ),
        )
        for expert_index, token_count in enumerate(expert_token_counts)
    )

    bytes_per_token = spec.hidden_size * 2
    transfer_rows = []
    for (source_die, destination_die), token_count in dispatch_counts.items():
        transfer_rows.append(
            (
                LiteMoeTransferRole.MOE_DISPATCH,
                source_die,
                destination_die,
                token_count,
            )
        )
        transfer_rows.append(
            (
                LiteMoeTransferRole.MOE_COMBINE,
                destination_die,
                source_die,
                token_count,
            )
        )
    role_rank = {
        LiteMoeTransferRole.MOE_DISPATCH: 0,
        LiteMoeTransferRole.MOE_COMBINE: 1,
    }
    p2p_metrics = tuple(
        LiteMoeP2PMetric(
            role=role,
            source_die_id=source_die,
            destination_die_id=destination_die,
            token_count=token_count,
            logical_bytes=token_count * bytes_per_token,
            hop_count=1,
            byte_hop_bytes=token_count * bytes_per_token,
        )
        for role, source_die, destination_die, token_count in sorted(
            transfer_rows,
            key=lambda row: (role_rank[row[0]], row[1], row[2]),
        )
    )

    oracle = LiteMoeOracle.create(
        case_id=spec.case_id,
        source_spec_id=spec.id,
        source_spec_digest=canonical_digest(spec),
        expert_metrics=expert_metrics,
        p2p_metrics=p2p_metrics,
        total_expert_gemm_flops=sum(
            metric.gemm_flops for metric in expert_metrics
        ),
        logical_p2p_bytes=sum(metric.logical_bytes for metric in p2p_metrics),
        per_hop_p2p_bytes=sum(
            metric.byte_hop_bytes for metric in p2p_metrics
        ),
    )
    oracle.validate_against(spec)
    return oracle


__all__ = ["build_lite_moe_oracle"]
