"""Frozen zero-argument production factory for the O1 Swizzle policy."""

from __future__ import annotations

from ..schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
)
from .swizzle_topo import SwizzlePlanner


SWIZZLE_POLICY_SCHEMA_VERSION = "wafer_frontend.swizzle_topo_policy/v1alpha1"


def production_swizzle_policy() -> SwizzlePlanner:
    hardware = SwizzleHardwareProfile.create(
        peak_flops_per_cycle=1024.0,
        confidence_fraction=0.1,
        efficiency_points=(SwizzleEfficiencyPoint(1, 1, 1, 0.75),),
        dte_launch_cycles=2,
        dte_sync_cycles=1,
        hop_latency_cycles=1,
        lane_bytes_per_cycle=32.0,
        max_inflight_dte=4,
        min_transfer_bytes=1,
        efficient_tile_floor=(1, 1, 1),
        sram_budget_bytes=16 * 1024 * 1024,
        double_buffer_supported=True,
    )
    constraints = SwizzleConstraints(
        allowed_algorithms=tuple(
            sorted(
                (
                    SwizzleAlgorithm.UNFUSED,
                    SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
                    SwizzleAlgorithm.MESHSLICE_2D_OS,
                ),
                key=lambda item: item.value,
            )
        ),
        max_candidates=32,
        max_actions=4096,
        max_buffers=512,
        max_chunk_count=64,
        allow_unroll_two=True,
    )
    return SwizzlePlanner(hardware, constraints)


__all__ = ["SWIZZLE_POLICY_SCHEMA_VERSION", "production_swizzle_policy"]
