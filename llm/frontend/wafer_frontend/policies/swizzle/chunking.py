"""Canonical rank-independent Wang chunk decomposition witnesses."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import SwizzleProblem, SwizzleSemanticWitness
from ...schema.swizzle import SwizzleTensorAxisRole


@dataclass(frozen=True, slots=True)
class WangChunkSpec:
    """One exact Wang decomposition point after all arithmetic gates."""

    chunk_count: int
    chunks_per_rank: int
    logical_bytes_per_chunk: int
    flops_per_rank_chunk: int

    def validate_against(
        self,
        problem: SwizzleProblem,
        semantic_witness: SwizzleSemanticWitness,
    ) -> None:
        _validate_inputs(problem, semantic_witness)
        ranks = len(problem.collective.participant_ranks)
        if (
            self.chunk_count not in (ranks, 2 * ranks, 4 * ranks, 8 * ranks)
            or self.chunk_count > problem.constraints.max_chunk_count
            or semantic_witness.split_axis.extent % self.chunk_count
            or self.chunk_count % ranks
            or self.chunks_per_rank != self.chunk_count // ranks
        ):
            raise SchemaError(
                "Wang chunk count is not a legal rank-multiple divisor",
                path="wang_chunk_spec.chunk_count",
            )
        if self.logical_bytes_per_chunk != wang_chunk_bytes(
            problem, self.chunk_count
        ):
            raise SchemaError(
                "Wang chunk payload does not conserve collective bytes",
                path="wang_chunk_spec.logical_bytes_per_chunk",
            )
        if self.flops_per_rank_chunk != wang_chunk_flops(
            problem, self.chunk_count
        ):
            raise SchemaError(
                "Wang chunk compute does not conserve GEMM FLOPs",
                path="wang_chunk_spec.flops_per_rank_chunk",
            )


def _validate_inputs(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
) -> None:
    problem.validate("swizzle_problem")
    semantic_witness.validate("semantic_witness")
    if semantic_witness.pattern is not problem.pattern:
        raise SchemaError(
            "semantic witness pattern mismatch",
            path="semantic_witness.pattern",
        )


def _collective_payload_bytes(problem: SwizzleProblem) -> int:
    ranks = len(problem.collective.participant_ranks)
    if problem.pattern is FusionPattern.AG_GEMM:
        return problem.collective.rank_input_bytes * ranks
    if problem.pattern is FusionPattern.GEMM_RS:
        return problem.collective.rank_output_bytes * ranks
    if problem.pattern is FusionPattern.GEMM_AR:
        return problem.collective.logical_bytes
    raise SchemaError(
        "unsupported Wang fusion pattern",
        path="swizzle_problem.pattern",
    )


def wang_chunk_bytes(problem: SwizzleProblem, chunk_count: int) -> int:
    """Return one transport payload while preserving the logical tensor."""

    problem.validate("swizzle_problem")
    payload = _collective_payload_bytes(problem)
    if type(chunk_count) is not int or chunk_count <= 0 or payload % chunk_count:
        raise SchemaError(
            "collective payload does not divide by Wang chunk count",
            path="chunk_count",
        )
    return payload // chunk_count


def wang_chunk_flops(problem: SwizzleProblem, chunk_count: int) -> int:
    """Return one rank/chunk COMP share with exact global FLOP closure."""

    problem.validate("swizzle_problem")
    ranks = len(problem.collective.participant_ranks)
    action_count = ranks * chunk_count
    if (
        type(chunk_count) is not int
        or chunk_count <= 0
        or problem.gemm.flops % action_count
    ):
        raise SchemaError(
            "GEMM FLOPs do not divide by rank/chunk actions",
            path="chunk_count",
        )
    return problem.gemm.flops // action_count


def wang_tile_shape(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
    chunk_count: int,
) -> tuple[int, int, int]:
    """Return the exact per-rank/per-chunk GEMM tile for one decomposition."""

    _validate_inputs(problem, semantic_witness)
    if (
        type(chunk_count) is not int
        or chunk_count <= 0
        or semantic_witness.split_axis.extent % chunk_count
    ):
        raise SchemaError(
            "split axis does not divide by Wang chunk count",
            path="chunk_count",
        )
    m, n, k = problem.gemm.m, problem.gemm.n, problem.gemm.k
    role = semantic_witness.split_axis.role
    if role is SwizzleTensorAxisRole.FREE_LHS:
        m //= chunk_count
    elif role is SwizzleTensorAxisRole.FREE_RHS:
        n //= chunk_count
    elif role is SwizzleTensorAxisRole.CONTRACT:
        k //= chunk_count
    return (m, n, k)


def legal_wang_chunk_specs(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
) -> tuple[WangChunkSpec, ...]:
    """Enumerate canonical ``divisors intersect {R,2R,4R,8R}`` specs.

    Arithmetic, minimum-transfer and configured chunk limits are applied before
    action materialization.  The fixed four-point family is already a hard
    canonical cap; duplicate values are nevertheless removed explicitly so the
    contract remains deterministic for future family extensions.
    """

    _validate_inputs(problem, semantic_witness)
    ranks = len(problem.collective.participant_ranks)
    if ranks < 2:
        return ()
    payload = _collective_payload_bytes(problem)
    candidates = tuple(dict.fromkeys((ranks, 2 * ranks, 4 * ranks, 8 * ranks)))
    result = []
    for chunk_count in candidates:
        if (
            chunk_count > problem.constraints.max_chunk_count
            or semantic_witness.split_axis.extent % chunk_count
            or payload % chunk_count
            or problem.gemm.flops % (ranks * chunk_count)
        ):
            continue
        chunk_bytes = payload // chunk_count
        if chunk_bytes < problem.hardware_profile.min_transfer_bytes:
            continue
        tile_shape = wang_tile_shape(problem, semantic_witness, chunk_count)
        if any(
            actual < required
            for actual, required in zip(
                tile_shape, problem.hardware_profile.efficient_tile_floor
            )
        ):
            continue
        spec = WangChunkSpec(
            chunk_count=chunk_count,
            chunks_per_rank=chunk_count // ranks,
            logical_bytes_per_chunk=chunk_bytes,
            flops_per_rank_chunk=problem.gemm.flops // (ranks * chunk_count),
        )
        spec.validate_against(problem, semantic_witness)
        result.append(spec)
    return tuple(result)


__all__ = [
    "WangChunkSpec",
    "legal_wang_chunk_specs",
    "wang_chunk_bytes",
    "wang_chunk_flops",
    "wang_tile_shape",
]
