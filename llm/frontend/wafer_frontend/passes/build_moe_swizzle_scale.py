"""Production construction of canonical C0-C4 MoE Swizzle workload truth.

This pass extends the validated LiteMoe DP4 C0 trace.  It does not construct
execution bytes/FLOPs in an integration runner and it does not widen the fixed
T8 LiteMoe execution implementation.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.lite_moe import LiteMoeStaticTrace, LiteMoeTraceAssignment
from ..schema.lite_moe_dp4 import (
    LiteMoeDp4Oracle,
    LiteMoeDp4Spec,
    LiteMoeDp4Topology,
)
from ..schema.serde import canonical_digest
from ..schema.swizzle_moe_scale import (
    MoeSwizzleExecutionStatus,
    MoeSwizzleScaleOracle,
    MoeSwizzleScaleRole,
    MoeSwizzleScaleSpec,
    MoeSwizzleTraceFamily,
)


MOE_SWIZZLE_SCALE_POINTS = (
    ("C0", 8, MoeSwizzleScaleRole.CONTROL, MoeSwizzleTraceFamily.BALANCED),
    ("C1", 32, MoeSwizzleScaleRole.CALIBRATION, MoeSwizzleTraceFamily.BALANCED),
    ("C2", 64, MoeSwizzleScaleRole.VALIDATION, MoeSwizzleTraceFamily.BALANCED),
    ("C3", 128, MoeSwizzleScaleRole.VALIDATION, MoeSwizzleTraceFamily.BALANCED),
    ("C4", 64, MoeSwizzleScaleRole.CAPACITY, MoeSwizzleTraceFamily.SKEWED),
)
MOE_SWIZZLE_C4_EXPERT_HISTOGRAM = (32, 16, 8, 8)


def _trace(
    name: str,
    tokens: int,
    family: MoeSwizzleTraceFamily,
    c0_spec: LiteMoeDp4Spec,
) -> LiteMoeStaticTrace:
    if name == "C0":
        return c0_spec.trace
    if family is MoeSwizzleTraceFamily.BALANCED:
        template = tuple(item.expert_index for item in c0_spec.trace.assignments)
        experts = tuple(template[index % len(template)] for index in range(tokens))
    else:
        experts = tuple(
            expert
            for expert, count in enumerate(MOE_SWIZZLE_C4_EXPERT_HISTOGRAM)
            for _ in range(count)
        )
        if len(experts) != tokens:
            raise SchemaError(
                "C4 histogram/token count drifted",
                path="moe_swizzle_scale_builder",
            )
    slots = [0] * c0_spec.expert_count
    assignments = []
    for token, expert in enumerate(experts):
        assignments.append(LiteMoeTraceAssignment(token, expert, slots[expert]))
        slots[expert] += 1
    return LiteMoeStaticTrace.create(
        token_count=tokens,
        assignments=tuple(assignments),
        expert_histogram=tuple(slots),
    )


def build_moe_swizzle_scale_truth(
    c0_spec: LiteMoeDp4Spec,
    c0_topology: LiteMoeDp4Topology,
) -> tuple[tuple[MoeSwizzleScaleSpec, MoeSwizzleScaleOracle], ...]:
    """Build typed scale truth only from validated production C0 inputs."""

    c0_spec.validate("c0_spec")
    c0_topology.validate_against(c0_spec, "c0_topology")
    result = []
    for name, tokens, role, family in MOE_SWIZZLE_SCALE_POINTS:
        trace = _trace(name, tokens, family, c0_spec)
        sources = tuple(
            c0_topology.token_source_die_ids[index % c0_spec.trace.token_count]
            for index in range(tokens)
        )
        spec = MoeSwizzleScaleSpec.create(
            name=name,
            role=role,
            trace_family=family,
            execution_status=(
                MoeSwizzleExecutionStatus.CAPACITY_PROBE
                if role is MoeSwizzleScaleRole.CAPACITY
                else MoeSwizzleExecutionStatus.PRODUCTION_READY
            ),
            source_c0_spec_id=c0_spec.id,
            source_c0_spec_digest=canonical_digest(c0_spec),
            source_c0_topology_id=c0_topology.id,
            source_c0_topology_digest=canonical_digest(c0_topology),
            tokens=tokens,
            hidden_size=c0_spec.hidden_size,
            intermediate_size=c0_spec.intermediate_size,
            expert_count=c0_spec.expert_count,
            top_k=c0_spec.top_k,
            capacity_per_expert=max(trace.expert_histogram),
            mesh_rows=2,
            mesh_columns=2,
            dtype=c0_spec.dtype,
            trace=trace,
            token_source_die_ids=sources,
            expert_home_die_ids=c0_topology.expert_home_die_ids,
        )
        spec.validate_against(c0_spec, c0_topology)
        result.append((spec, MoeSwizzleScaleOracle.create(spec)))
    if tuple(spec.name for spec, _ in result) != tuple(
        item[0] for item in MOE_SWIZZLE_SCALE_POINTS
    ):
        raise SchemaError(
            "C0-C4 canonical order drifted",
            path="moe_swizzle_scale_builder",
        )
    return tuple(result)


def validate_moe_swizzle_c0_oracle(
    scale_oracle: MoeSwizzleScaleOracle,
    c0_oracle: LiteMoeDp4Oracle,
    c0_spec: LiteMoeDp4Spec,
    c0_topology: LiteMoeDp4Topology,
) -> None:
    """Prove the new C0 work oracle is exactly the legacy production oracle."""

    c0_oracle.validate_against(c0_spec, c0_topology, "c0_oracle")
    scale_oracle.validate()
    actual = (
        scale_oracle.expert_token_counts,
        scale_oracle.expert_gemm_flops,
        scale_oracle.remote_token_indices,
        scale_oracle.dispatch_logical_bytes,
        scale_oracle.combine_logical_bytes,
        scale_oracle.total_expert_gemm_flops,
        scale_oracle.logical_p2p_bytes,
        scale_oracle.data_packets,
    )
    expected = (
        c0_oracle.expert_token_counts,
        c0_oracle.expert_gemm_flops,
        c0_oracle.remote_token_indices,
        c0_oracle.dispatch_logical_bytes,
        c0_oracle.combine_logical_bytes,
        c0_oracle.total_expert_gemm_flops,
        c0_oracle.logical_p2p_bytes,
        c0_oracle.data_packets,
    )
    if actual != expected:
        raise SchemaError(
            "C0 scale oracle does not equal legacy production truth",
            path="scale_oracle",
        )


__all__ = [
    "MOE_SWIZZLE_C4_EXPERT_HISTOGRAM",
    "MOE_SWIZZLE_SCALE_POINTS",
    "build_moe_swizzle_scale_truth",
    "validate_moe_swizzle_c0_oracle",
]
