"""Two-stage row/column personalized MoE candidate enumeration."""

from __future__ import annotations

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import SwizzleAlgorithm
from ...schema.swizzle_moe_calibration import MoeSwizzleCalibrationProfile
from .moe_direct_xy import (
    _COMBINE_TRANSPORT_BLOCK_COUNTS,
    _candidate_token_block_sizes,
    _endpoints,
    _finish_candidate,
    _packet,
    _remote_runs,
    _validate_sources,
    _whole_offsets,
)


def _build_comet(
    problem,
    spec,
    execution,
    *,
    token_block_size,
    transport_output_block_count,
    unroll_degree,
    double_buffer,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    pattern = problem.region.pattern
    algorithm = SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A
    pivots = {
        (source, destination): pivot
        for source, destination, pivot in problem.topology.pivot_by_pair
    }
    routes = {
        (item.source_rank, item.destination_rank): item
        for item in problem.topology.group.routes
    }
    packets = []
    first_stage_packets = set()
    predecessor_by_packet = {}
    final_packet_by_block = {}
    pivot_bytes = 0
    for assignments in _remote_runs(problem, token_block_size, algorithm):
        source, destination, _ = _endpoints(assignments[0], pattern)
        pivot = pivots[(source, destination)]
        n_blocks = (
            (None,)
            if (
                pattern is FusionPattern.MOE_DISPATCH_GEMM
                or transport_output_block_count == 1
            )
            else tuple(range(transport_output_block_count))
        )
        for n_block in n_blocks:
            extent = (
                None
                if n_block is None
                else spec.hidden_size * 2 // transport_output_block_count
            )
            block_extent = assignments[0].payload_bytes if extent is None else extent
            final_offsets = {
                item.id: _whole_offsets(item, pattern)[1]
                + (0 if n_block is None else n_block * block_extent)
                for item in assignments
            }
            if pivot == destination:
                packet = _packet(
                    assignments, pattern, stage=0, source_rank=source,
                    destination_rank=destination, pivot_rank=pivot,
                    route_ref=routes[(source, destination)].id,
                    n_block_index=n_block, slice_bytes=extent,
                )
                packets.append(packet)
                first_stage_packets.add(packet.id)
                final = packet
            elif pivot == source:
                packet = _packet(
                    assignments, pattern, stage=1, source_rank=source,
                    destination_rank=destination, pivot_rank=pivot,
                    route_ref=routes[(source, destination)].id,
                    n_block_index=n_block, slice_bytes=extent,
                )
                packets.append(packet)
                first_stage_packets.add(packet.id)
                final = packet
            else:
                stage0 = _packet(
                    assignments, pattern, stage=0, source_rank=source,
                    destination_rank=pivot, pivot_rank=pivot,
                    route_ref=routes[(source, pivot)].id,
                    n_block_index=n_block, slice_bytes=extent,
                    destination_offsets=lambda item, offsets=final_offsets: offsets[item.id],
                )
                stage1 = _packet(
                    assignments, pattern, stage=1, source_rank=pivot,
                    destination_rank=destination, pivot_rank=pivot,
                    route_ref=routes[(pivot, destination)].id,
                    n_block_index=n_block, slice_bytes=extent,
                    source_offsets=lambda item, offsets=final_offsets: offsets[item.id],
                    destination_offsets=lambda item, offsets=final_offsets: offsets[item.id],
                )
                packets.extend((stage0, stage1))
                first_stage_packets.add(stage0.id)
                predecessor_by_packet[stage1.id] = stage0.id
                pivot_bytes += stage0.logical_bytes
                final = stage1
            for assignment in assignments:
                if n_block is None and pattern is FusionPattern.MOE_GEMM_COMBINE:
                    final_packet_by_block[(assignment.id, 0)] = final.id
                else:
                    final_packet_by_block[(assignment.id, n_block)] = final.id
    physical_slots = 2 if double_buffer else 1
    if pivot_bytes * physical_slots > problem.sram_capacity_bytes:
        raise SchemaError("Comet pivot SRAM exceeds typed capacity", path="moe_comet_mesh.sram")
    return _finish_candidate(
        problem,
        spec,
        execution,
        algorithm,
        tuple(packets),
        final_packet_by_block,
        first_stage_packets,
        predecessor_by_packet,
        token_block_size=token_block_size,
        compute_output_block_count=(
            1
        ),
        transport_output_block_count=transport_output_block_count,
        unroll_degree=unroll_degree,
        double_buffer=double_buffer,
        calibration_profile=calibration_profile,
    )


def build_comet_mesh_moe_candidates(
    problem, spec, oracle, execution, *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    _validate_sources(problem, spec, oracle, execution, "moe_comet_mesh")
    algorithm = SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A
    if algorithm not in problem.allowed_algorithms or not problem.topology.complete_rectangle:
        raise SchemaError("Comet mesh requires an admitted complete rectangle", path="moe_comet_mesh")
    token_block_size = _candidate_token_block_sizes(spec)[-1]
    transport_output_block_count = (
        1 if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM else 2
    )
    candidates = tuple(
        _build_comet(
            problem,
            spec,
            execution,
            token_block_size=token_block_size,
            transport_output_block_count=transport_output_block_count,
            unroll_degree=unroll,
            double_buffer=double_buffer,
            calibration_profile=calibration_profile,
        )
        for unroll, double_buffer in ((1, False), (2, True))
    )
    if len({item.id for item in candidates}) != 2:
        raise SchemaError("Comet variants are not distinct", path="moe_comet_mesh")
    return candidates


def build_comet_mesh_moe_candidate_grid(
    problem, spec, oracle, execution, *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    _validate_sources(problem, spec, oracle, execution, "moe_comet_mesh")
    algorithm = SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A
    if algorithm not in problem.allowed_algorithms or not problem.topology.complete_rectangle:
        raise SchemaError("Comet mesh requires an admitted complete rectangle", path="moe_comet_mesh")
    transport_output_block_counts = (
        (1,)
        if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
        else tuple(reversed(_COMBINE_TRANSPORT_BLOCK_COUNTS))
    )
    candidates = tuple(
        _build_comet(
            problem,
            spec,
            execution,
            token_block_size=token_block_size,
            transport_output_block_count=transport_output_block_count,
            unroll_degree=unroll,
            double_buffer=double_buffer,
            calibration_profile=calibration_profile,
        )
        for transport_output_block_count in transport_output_block_counts
        for token_block_size in reversed(_candidate_token_block_sizes(spec))
        for unroll, double_buffer in ((1, False), (2, True))
    )
    if len({item.id for item in candidates}) != len(candidates):
        raise SchemaError("Comet M-block/unroll variants are not distinct", path="moe_comet_mesh")
    return candidates


def build_comet_mesh_moe_candidate(
    problem, spec, oracle, execution, *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    return build_comet_mesh_moe_candidates(
        problem, spec, oracle, execution, calibration_profile=calibration_profile,
    )[0]


__all__ = [
    "build_comet_mesh_moe_candidate",
    "build_comet_mesh_moe_candidate_grid",
    "build_comet_mesh_moe_candidates",
]
