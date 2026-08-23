"""Build the exact sequential collective/GEMM W10 baseline."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import DType, MeshAxisName, stable_artifact_id
from ..schema.ir0 import FusionPattern
from ..schema.ir1 import IR1
from ..schema.swizzle import SwizzleActionKind, SwizzleAlgorithm, SwizzleCandidate, SwizzleOperand, SwizzleProblem, SwizzleSemanticWitness
from ..schema.swizzle_plan import SwizzleValueUse
from ..schema.swizzle_unfused import (
    UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION,
    UnfusedComparisonAction,
    UnfusedComparisonFlow,
    UnfusedComparisonOperand,
    UnfusedComparisonPlan,
    UnfusedComparisonProjection,
    UnfusedComparisonRankProgram,
    UnfusedComparisonRankProjection,
    UnfusedComparisonStage,
)


@dataclass(frozen=True, slots=True)
class _ValueSpec:
    ref: str
    source_tensor_ref: str
    shape: tuple[int, ...]
    layout: str
    dtype: DType
    storage_ref: str = ""
    storage_bytes: int = 0
    byte_offset: int = 0
    tensor_offset: tuple[int, ...] = ()

    @property
    def bytes(self) -> int:
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements * (2 if self.dtype is DType.FP16 else 4)


def _value(problem: SwizzleProblem, rank: int, name: str) -> str:
    return stable_artifact_id(
        "unfused_comparison_value",
        {"problem": problem.id, "rank": rank, "name": name},
        schema_version=UNFUSED_COMPARISON_PROJECTION_SCHEMA_VERSION,
    )


def _route(problem: SwizzleProblem, source: int, destination: int):
    matches = tuple(
        item
        for item in problem.group.routes
        if (item.source_rank, item.destination_rank) == (source, destination)
    )
    if len(matches) != 1:
        raise SchemaError(
            "UNFUSED V1 requires one exact directed route per rank pair",
            path="problem.group.routes",
        )
    return matches[0]


def _action(
    *,
    rank: int,
    kind: SwizzleActionKind,
    stage: UnfusedComparisonStage,
    member_ref: str,
    deps: tuple[str, ...] = (),
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    peer_rank: int | None = None,
    route=None,
    logical_bytes: int = 0,
    flops: int = 0,
) -> UnfusedComparisonAction:
    return UnfusedComparisonAction.create(
        rank=rank,
        kind=kind,
        stage=stage,
        member_ref=member_ref,
        deps=deps,
        read_value_refs=reads,
        write_value_refs=writes,
        peer_rank=peer_rank,
        route_ref=route.id if route is not None else None,
        die_path=route.die_path if route is not None else (),
        logical_bytes=logical_bytes,
        flops=flops,
    )


def _rank_local_view(
    problem: SwizzleProblem,
    view,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Rebuild one exact TP-rank tensor view from typed sharding facts."""

    mapped = tuple(
        index
        for index, axis in enumerate(view.sharding_dim_map)
        if axis is MeshAxisName.TP
    )
    if len(mapped) > 1 or any(
        axis not in (None, MeshAxisName.TP)
        for axis in view.sharding_dim_map
    ):
        raise SchemaError(
            "UNFUSED rank-local view supports one exact TP-sharded dimension",
            path="problem.gemm",
        )
    ranks = tuple(item.rank for item in problem.group.placements)
    if rank not in ranks:
        raise SchemaError(
            "rank is absent from the exact group placement",
            path="problem.group.placements",
        )
    shape = list(view.shape)
    offset = [0] * len(shape)
    if mapped:
        axis = mapped[0]
        degree = len(ranks)
        if shape[axis] % degree:
            raise SchemaError(
                "TP-sharded tensor extent must divide the exact group degree",
                path=f"problem.gemm.sharding_dim_map[{axis}]",
            )
        extent = shape[axis] // degree
        coordinate = ranks.index(rank)
        shape[axis] = extent
        offset[axis] = coordinate * extent
    return tuple(shape), tuple(offset)


def _gemm_values(problem: SwizzleProblem, rank: int) -> tuple[_ValueSpec, _ValueSpec, _ValueSpec]:
    gemm = problem.gemm
    result = []
    for name, view in (
        ("gemm_lhs", gemm.lhs),
        ("gemm_rhs", gemm.rhs),
        ("gemm_output", gemm.output),
    ):
        shape, tensor_offset = _rank_local_view(problem, view, rank)
        result.append(_ValueSpec(
            _value(problem, rank, name),
            view.value_ref,
            shape,
            view.layout,
            gemm.dtype,
            tensor_offset=tensor_offset,
        ))
    return tuple(result)


def _gemm_flops(
    lhs: _ValueSpec,
    rhs: _ValueSpec,
    output: _ValueSpec,
) -> int:
    if (
        len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or len(output.shape) != 2
        or lhs.shape[1] != rhs.shape[0]
        or output.shape != (lhs.shape[0], rhs.shape[1])
    ):
        raise SchemaError(
            "rank-local GEMM views do not form one exact matrix product",
            path="problem.gemm",
        )
    return 2 * lhs.shape[0] * lhs.shape[1] * rhs.shape[1]


def _local_shard_offset(
    problem: SwizzleProblem,
    view,
    rank: int,
) -> tuple[int, ...]:
    """Locate a descriptor that is already rank-local in its global tensor."""

    mapped = tuple(
        index
        for index, axis in enumerate(view.sharding_dim_map)
        if axis is MeshAxisName.TP
    )
    if len(mapped) > 1:
        raise SchemaError(
            "rank-local output supports one exact TP-sharded dimension",
            path="problem.collective.output.sharding_dim_map",
        )
    offset = [0] * len(view.shape)
    if mapped:
        ranks = tuple(item.rank for item in problem.group.placements)
        if rank not in ranks:
            raise SchemaError(
                "rank is absent from the exact group placement",
                path="problem.group.placements",
            )
        axis = mapped[0]
        offset[axis] = ranks.index(rank) * view.shape[axis]
    return tuple(offset)


def _build_multi_rank_actions(
    problem: SwizzleProblem,
    semantic: SwizzleSemanticWitness,
) -> tuple[tuple[UnfusedComparisonAction, ...], dict[str, _ValueSpec]]:
    """Build the direct all-pairs AG/RS baseline for more than two ranks."""

    coll = problem.collective
    ranks = coll.participant_ranks
    actions_by_rank = {rank: [] for rank in ranks}
    values: dict[str, _ValueSpec] = {}
    gemm_values = {}
    for rank in ranks:
        lhs, rhs, output = _gemm_values(problem, rank)
        gemm_values[rank] = (lhs, rhs, output)
        values.update((item.ref, item) for item in (lhs, rhs, output))

    if problem.pattern is FusionPattern.AG_GEMM:
        local_values = {}
        received_values = {}
        gathered_values = {}
        sends = {}
        for rank in ranks:
            storage = _value(problem, rank, "collective_output_storage")
            local = _ValueSpec(
                _value(problem, rank, "collective_input"),
                coll.input.value_ref,
                coll.input.shape,
                coll.input.layout,
                problem.gemm.dtype,
                storage,
                coll.rank_output_bytes,
                ranks.index(rank) * coll.rank_input_bytes,
            )
            gathered = _ValueSpec(
                _value(problem, rank, "collective_output"),
                coll.output.value_ref,
                coll.output.shape,
                coll.output.layout,
                problem.gemm.dtype,
                storage,
                coll.rank_output_bytes,
                0,
            )
            if local.bytes != coll.rank_input_bytes or gathered.bytes != coll.rank_output_bytes:
                raise SchemaError(
                    "AG typed views disagree with collective bytes",
                    path="problem.collective",
                )
            local_values[rank] = local
            gathered_values[rank] = gathered
            values.update((item.ref, item) for item in (local, gathered))
            for source in ranks:
                if source == rank:
                    continue
                received = _ValueSpec(
                    _value(problem, rank, f"received_shard_from_rank_{source}"),
                    coll.input.value_ref,
                    coll.input.shape,
                    coll.input.layout,
                    problem.gemm.dtype,
                    storage,
                    coll.rank_output_bytes,
                    ranks.index(source) * coll.rank_input_bytes,
                )
                if received.bytes != coll.rank_input_bytes:
                    raise SchemaError(
                        "AG typed views disagree with collective bytes",
                        path="problem.collective",
                    )
                received_values[(rank, source)] = received
                values[received.ref] = received

        if len(ranks) != 4:
            raise SchemaError(
                "multi-rank UNFUSED AG requires the exact four-rank wave schedule",
                path="problem.collective.participant_ranks",
            )
        waits_by_rank = {rank: [] for rank in ranks}
        for partner_mask in (1, 2, 3):
            for source_index, source in enumerate(ranks):
                destination = ranks[source_index ^ partner_mask]
                previous_wait = (
                    waits_by_rank[source][-1]
                    if waits_by_rank[source]
                    else None
                )
                send = _action(
                    rank=source,
                    kind=SwizzleActionKind.SEND,
                    stage=UnfusedComparisonStage.COLLECTIVE,
                    member_ref=coll.node_ref,
                    deps=(previous_wait.id,) if previous_wait is not None else (),
                    reads=(local_values[source].ref,),
                    peer_rank=destination,
                    route=_route(problem, source, destination),
                    logical_bytes=coll.rank_input_bytes,
                )
                sends[(source, destination)] = send
                actions_by_rank[source].append(send)

            for destination_index, rank in enumerate(ranks):
                source = ranks[destination_index ^ partner_mask]
                received = received_values[(rank, source)]
                previous_wait = (
                    waits_by_rank[rank][-1]
                    if waits_by_rank[rank]
                    else None
                )
                recv_deps = (sends[(source, rank)].id,)
                if previous_wait is not None:
                    recv_deps += (previous_wait.id,)
                recv = _action(
                    rank=rank,
                    kind=SwizzleActionKind.RECV,
                    stage=UnfusedComparisonStage.COLLECTIVE,
                    member_ref=coll.node_ref,
                    deps=recv_deps,
                    writes=(received.ref,),
                    peer_rank=source,
                    route=_route(problem, source, rank),
                    logical_bytes=coll.rank_input_bytes,
                )
                wait = _action(
                    rank=rank,
                    kind=SwizzleActionKind.WAIT,
                    stage=UnfusedComparisonStage.COMPLETE,
                    member_ref=coll.node_ref,
                    deps=(recv.id,),
                    reads=(received.ref,),
                )
                actions_by_rank[rank].extend((recv, wait))
                waits_by_rank[rank].append(wait)

        for rank in ranks:
            waits = waits_by_rank[rank]
            lhs, rhs, output = gemm_values[rank]
            inputs = (
                (gathered_values[rank].ref, rhs.ref)
                if semantic.gemm_operand is SwizzleOperand.LHS
                else (lhs.ref, gathered_values[rank].ref)
            )
            local_sends = tuple(
                sends[(rank, destination)].id
                for destination in ranks
                if destination != rank
            )
            comp = _action(
                rank=rank,
                kind=SwizzleActionKind.COMP,
                stage=UnfusedComparisonStage.GEMM,
                member_ref=problem.gemm.node_ref,
                deps=local_sends + tuple(wait.id for wait in waits),
                reads=inputs,
                writes=(output.ref,),
                flops=_gemm_flops(lhs, rhs, output),
            )
            actions_by_rank[rank].append(comp)
        return tuple(tuple(actions_by_rank[rank]) for rank in ranks), values

    if problem.pattern is not FusionPattern.GEMM_RS:
        raise SchemaError(
            "multi-rank UNFUSED comparison supports AG and RS only",
            path="problem.pattern",
        )

    dtype_bytes = 2 if problem.gemm.dtype is DType.FP16 else 4
    chunk_bytes = coll.rank_output_bytes
    if chunk_bytes % dtype_bytes:
        raise SchemaError(
            "collective shard bytes are not dtype aligned",
            path="problem.collective.rank_output_bytes",
        )
    gemms = {}
    sends = {}
    send_chunks = {}
    packings = {}
    received_values = {}
    accumulator_values = {}
    reduced_values = {}
    for rank in ranks:
        lhs, rhs, output = gemm_values[rank]
        gemm = _action(
            rank=rank,
            kind=SwizzleActionKind.COMP,
            stage=UnfusedComparisonStage.GEMM,
            member_ref=problem.gemm.node_ref,
            reads=(lhs.ref, rhs.ref),
            writes=(output.ref,),
            flops=_gemm_flops(lhs, rhs, output),
        )
        gemms[rank] = gemm
        actions_by_rank[rank].append(gemm)

        local_chunk = _ValueSpec(
            _value(problem, rank, "local_chunk"),
            coll.input.value_ref,
            coll.output.shape,
            coll.output.layout,
            problem.gemm.dtype,
            output.ref,
            coll.rank_input_bytes,
            ranks.index(rank) * chunk_bytes,
        )
        received_storage = _value(problem, rank, "received_reduce_storage")
        terminal_storage = _value(problem, rank, "terminal_reduce_storage")
        received = _ValueSpec(
            _value(problem, rank, "received_contribution"),
            coll.input.value_ref,
            coll.output.shape,
            coll.output.layout,
            problem.gemm.dtype,
            received_storage,
            chunk_bytes,
            0,
        )
        accumulator = _ValueSpec(
            _value(problem, rank, "packed_local_chunk"),
            coll.input.value_ref,
            coll.output.shape,
            coll.output.layout,
            problem.gemm.dtype,
            terminal_storage,
            chunk_bytes,
            0,
        )
        reduced = _ValueSpec(
            _value(problem, rank, "reduced_chunk"),
            coll.output.value_ref,
            coll.output.shape,
            coll.output.layout,
            problem.gemm.dtype,
            terminal_storage,
            chunk_bytes,
            0,
            _local_shard_offset(problem, coll.output, rank),
        )
        if any(item.bytes != chunk_bytes for item in (local_chunk, received, accumulator, reduced)):
            raise SchemaError(
                "RS typed views disagree with collective bytes",
                path="problem.collective",
            )
        received_values[rank] = received
        accumulator_values[rank] = accumulator
        reduced_values[rank] = reduced
        values.update(
            (item.ref, item)
            for item in (local_chunk, received, accumulator, reduced)
        )
        for owner in ranks:
            if owner == rank:
                continue
            send_chunk = _ValueSpec(
                _value(problem, rank, f"send_chunk_for_rank_{owner}"),
                coll.input.value_ref,
                coll.output.shape,
                coll.output.layout,
                problem.gemm.dtype,
                output.ref,
                coll.rank_input_bytes,
                ranks.index(owner) * chunk_bytes,
            )
            if send_chunk.bytes != chunk_bytes:
                raise SchemaError(
                    "RS typed views disagree with collective bytes",
                    path="problem.collective",
                )
            values[send_chunk.ref] = send_chunk
            send_chunks[(rank, owner)] = send_chunk
        packing = _action(
            rank=rank,
            kind=SwizzleActionKind.LOCAL_COPY,
            stage=UnfusedComparisonStage.REDUCTION,
            member_ref=coll.node_ref,
            deps=(gemm.id,),
            reads=(local_chunk.ref,),
            writes=(accumulator.ref,),
            logical_bytes=chunk_bytes,
        )
        packings[rank] = packing
        actions_by_rank[rank].append(packing)

    if len(ranks) != 4:
        raise SchemaError(
            "multi-rank UNFUSED RS requires the exact four-rank wave schedule",
            path="problem.collective.participant_ranks",
        )
    previous_reduce = {rank: None for rank in ranks}
    for partner_mask in (1, 2, 3):
        for rank_index, rank in enumerate(ranks):
            peer = ranks[rank_index ^ partner_mask]
            prior = previous_reduce[rank]
            send = _action(
                rank=rank,
                kind=SwizzleActionKind.SEND,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref,
                deps=(prior.id,) if prior is not None else (gemms[rank].id,),
                reads=(send_chunks[(rank, peer)].ref,),
                peer_rank=peer,
                route=_route(problem, rank, peer),
                logical_bytes=chunk_bytes,
            )
            sends[(rank, peer)] = send
            actions_by_rank[rank].append(send)

        for rank_index, rank in enumerate(ranks):
            peer = ranks[rank_index ^ partner_mask]
            prior = previous_reduce[rank]
            recv_deps = (sends[(peer, rank)].id,)
            if prior is not None:
                recv_deps += (prior.id,)
            recv = _action(
                rank=rank,
                kind=SwizzleActionKind.RECV,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref,
                deps=recv_deps,
                writes=(received_values[rank].ref,),
                peer_rank=peer,
                route=_route(problem, peer, rank),
                logical_bytes=chunk_bytes,
            )
            wait = _action(
                rank=rank,
                kind=SwizzleActionKind.WAIT,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref,
                deps=(recv.id,),
                reads=(received_values[rank].ref,),
            )
            if prior is None:
                reduce_deps = (
                    gemms[rank].id,
                    packings[rank].id,
                    sends[(rank, peer)].id,
                    wait.id,
                )
                accumulator_ref = accumulator_values[rank].ref
            else:
                reduce_deps = (
                    prior.id,
                    sends[(rank, peer)].id,
                    wait.id,
                )
                accumulator_ref = reduced_values[rank].ref
            reduction = _action(
                rank=rank,
                kind=SwizzleActionKind.REDUCE,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref,
                deps=reduce_deps,
                reads=(received_values[rank].ref, accumulator_ref),
                writes=(reduced_values[rank].ref,),
            )
            actions_by_rank[rank].extend((recv, wait, reduction))
            previous_reduce[rank] = reduction
    return tuple(tuple(actions_by_rank[rank]) for rank in ranks), values


def _build_actions(problem: SwizzleProblem, semantic: SwizzleSemanticWitness) -> tuple[tuple[UnfusedComparisonAction, ...], dict[str, _ValueSpec]]:
    coll = problem.collective
    ranks = coll.participant_ranks
    placement_ranks = tuple(item.rank for item in problem.group.placements)
    if ranks != placement_ranks:
        raise SchemaError(
            "UNFUSED comparison requires every canonical placed rank",
            path="problem.collective.participant_ranks",
        )
    if len(ranks) != 2:
        return _build_multi_rank_actions(problem, semantic)
    actions_by_rank: list[list[UnfusedComparisonAction]] = [[], []]
    values: dict[str, _ValueSpec] = {}
    gemm_values = {}
    for rank in (0, 1):
        lhs, rhs, output = _gemm_values(problem, rank)
        gemm_values[rank] = (lhs, rhs, output)
        values.update((item.ref, item) for item in (lhs, rhs, output))

    if problem.pattern is FusionPattern.AG_GEMM:
        sends = {}
        recv_values = {}
        gathered_values = {}
        for rank in (0, 1):
            peer = 1 - rank
            storage = _value(problem, rank, "collective_output_storage")
            local = _ValueSpec(
                _value(problem, rank, "collective_input"), coll.input.value_ref,
                coll.input.shape, coll.input.layout, problem.gemm.dtype,
                storage, coll.rank_output_bytes, rank * coll.rank_input_bytes,
            )
            received = _ValueSpec(
                _value(problem, rank, "received_shard"), coll.input.value_ref,
                coll.input.shape, coll.input.layout, problem.gemm.dtype,
                storage, coll.rank_output_bytes, peer * coll.rank_input_bytes,
            )
            gathered = _ValueSpec(
                _value(problem, rank, "collective_output"), coll.output.value_ref,
                coll.output.shape, coll.output.layout, problem.gemm.dtype,
                storage, coll.rank_output_bytes, 0,
            )
            if local.bytes != coll.rank_input_bytes or received.bytes != coll.rank_input_bytes or gathered.bytes != coll.rank_output_bytes:
                raise SchemaError("AG typed views disagree with collective bytes", path="problem.collective")
            values.update((item.ref, item) for item in (local, received, gathered))
            recv_values[rank], gathered_values[rank] = received, gathered
            route = _route(problem, rank, peer)
            sends[rank] = _action(
                rank=rank, kind=SwizzleActionKind.SEND,
                stage=UnfusedComparisonStage.COLLECTIVE,
                member_ref=coll.node_ref, reads=(local.ref,), peer_rank=peer,
                route=route, logical_bytes=coll.rank_input_bytes,
            )
            actions_by_rank[rank].append(sends[rank])
        for rank in (0, 1):
            peer = 1 - rank
            recv = _action(
                rank=rank, kind=SwizzleActionKind.RECV,
                stage=UnfusedComparisonStage.COLLECTIVE,
                member_ref=coll.node_ref, deps=(sends[peer].id,),
                writes=(recv_values[rank].ref,), peer_rank=peer,
                route=_route(problem, peer, rank), logical_bytes=coll.rank_input_bytes,
            )
            wait = _action(
                rank=rank, kind=SwizzleActionKind.WAIT,
                stage=UnfusedComparisonStage.COMPLETE,
                member_ref=coll.node_ref, deps=(recv.id,),
                reads=(_value(problem, rank, "collective_input"), recv_values[rank].ref),
            )
            lhs, rhs, output = gemm_values[rank]
            inputs = (gathered_values[rank].ref, rhs.ref) if semantic.gemm_operand is SwizzleOperand.LHS else (lhs.ref, gathered_values[rank].ref)
            comp = _action(
                rank=rank, kind=SwizzleActionKind.COMP,
                stage=UnfusedComparisonStage.GEMM,
                member_ref=problem.gemm.node_ref,
                deps=(sends[rank].id, wait.id), reads=inputs,
                writes=(output.ref,), flops=_gemm_flops(lhs, rhs, output),
            )
            actions_by_rank[rank].extend((recv, wait, comp))
    else:
        gemms = {}
        sends = {}
        received = {}
        local_chunks = {}
        send_chunks = {}
        packed_chunks = {}
        packings = {}
        reduced = {}
        dtype_bytes = 2 if problem.gemm.dtype is DType.FP16 else 4
        chunk_bytes = (
            coll.rank_output_bytes
            if problem.pattern is FusionPattern.GEMM_RS
            else coll.rank_output_bytes // 2
        )
        if coll.rank_output_bytes % (dtype_bytes * (2 if problem.pattern is FusionPattern.GEMM_AR else 1)):
            raise SchemaError("collective shard bytes are not dtype aligned", path="problem.collective.rank_output_bytes")
        chunk_shape = (chunk_bytes // dtype_bytes,)
        for rank in (0, 1):
            peer = 1 - rank
            lhs, rhs, output = gemm_values[rank]
            gemms[rank] = _action(
                rank=rank, kind=SwizzleActionKind.COMP,
                stage=UnfusedComparisonStage.GEMM,
                member_ref=problem.gemm.node_ref,
                reads=(lhs.ref, rhs.ref), writes=(output.ref,), flops=_gemm_flops(lhs, rhs, output),
            )
            actions_by_rank[rank].append(gemms[rank])
            shape = coll.output.shape if problem.pattern is FusionPattern.GEMM_RS else chunk_shape
            layout = coll.output.layout if problem.pattern is FusionPattern.GEMM_RS else "flat_transport/v1"
            gemm_storage = output.ref
            local_chunks[rank] = _ValueSpec(_value(problem, rank, "local_chunk"), coll.input.value_ref, shape, layout, problem.gemm.dtype, gemm_storage, coll.rank_input_bytes, rank * chunk_bytes)
            send_chunks[rank] = _ValueSpec(_value(problem, rank, "send_chunk"), coll.input.value_ref, shape, layout, problem.gemm.dtype, gemm_storage, coll.rank_input_bytes, peer * chunk_bytes)
            reduce_storage = _value(problem, rank, "contiguous_reduce_storage")
            if problem.pattern is FusionPattern.GEMM_AR:
                storage_bytes = 3 * chunk_bytes
                packed_offset = rank * chunk_bytes
                reduced_offset = (rank + 1) * chunk_bytes
            else:
                reduce_storage = _value(problem, rank, "packed_reduce_storage")
                terminal_storage = _value(problem, rank, "terminal_reduce_storage")
                storage_bytes = chunk_bytes
                packed_offset = 0
                reduced_offset = 0
            packed_chunks[rank] = _ValueSpec(
                _value(problem, rank, "packed_local_chunk"), coll.input.value_ref,
                shape, layout, problem.gemm.dtype,
                reduce_storage, storage_bytes, packed_offset,
            )
            received[rank] = _ValueSpec(
                _value(problem, rank, "received_contribution"), coll.input.value_ref,
                shape, layout, problem.gemm.dtype,
                (
                    reduce_storage
                    if problem.pattern is FusionPattern.GEMM_AR
                    else terminal_storage
                ),
                storage_bytes, reduced_offset,
            )
            reduced[rank] = _ValueSpec(
                _value(problem, rank, "reduced_chunk"), coll.output.value_ref,
                shape, layout, problem.gemm.dtype,
                (
                    reduce_storage
                    if problem.pattern is FusionPattern.GEMM_AR
                    else terminal_storage
                ),
                storage_bytes, reduced_offset,
                (
                    _local_shard_offset(problem, coll.output, rank)
                    if problem.pattern is FusionPattern.GEMM_RS
                    else (0,) * len(shape)
                ),
            )
            values.update((item.ref, item) for item in (
                local_chunks[rank], send_chunks[rank], packed_chunks[rank],
                received[rank], reduced[rank],
            ))
            sends[rank] = _action(
                rank=rank, kind=SwizzleActionKind.SEND,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref, deps=(gemms[rank].id,),
                reads=(send_chunks[rank].ref,), peer_rank=peer,
                route=_route(problem, rank, peer), logical_bytes=chunk_bytes,
            )
            actions_by_rank[rank].append(sends[rank])
            packings[rank] = _action(
                rank=rank, kind=SwizzleActionKind.LOCAL_COPY,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref, deps=(gemms[rank].id,),
                reads=(local_chunks[rank].ref,),
                writes=(packed_chunks[rank].ref,),
                logical_bytes=chunk_bytes,
            )
            actions_by_rank[rank].append(packings[rank])
        reductions = {}
        for rank in (0, 1):
            peer = 1 - rank
            recv = _action(
                rank=rank, kind=SwizzleActionKind.RECV,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref, deps=(sends[peer].id,),
                writes=(received[rank].ref,), peer_rank=peer,
                route=_route(problem, peer, rank), logical_bytes=chunk_bytes,
            )
            wait = _action(
                rank=rank, kind=SwizzleActionKind.WAIT,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref, deps=(recv.id,), reads=(received[rank].ref,),
            )
            reductions[rank] = _action(
                rank=rank, kind=SwizzleActionKind.REDUCE,
                stage=UnfusedComparisonStage.REDUCTION,
                member_ref=coll.node_ref,
                deps=(gemms[rank].id, sends[rank].id, packings[rank].id, wait.id),
                reads=(packed_chunks[rank].ref, received[rank].ref),
                writes=(reduced[rank].ref,),
            )
            actions_by_rank[rank].extend((recv, wait, reductions[rank]))
        if problem.pattern is FusionPattern.GEMM_AR:
            rep_sends = {}
            peer_chunks = {}
            outputs = {}
            for rank in (0, 1):
                peer = 1 - rank
                peer_chunks[rank] = _ValueSpec(
                    _value(problem, rank, "replicated_peer_chunk"), coll.output.value_ref,
                    chunk_shape, "flat_transport/v1", problem.gemm.dtype,
                    reduced[rank].storage_ref, reduced[rank].storage_bytes,
                    (peer + 1) * chunk_bytes,
                )
                outputs[rank] = _ValueSpec(
                    _value(problem, rank, "collective_output"), coll.output.value_ref,
                    coll.output.shape, coll.output.layout, problem.gemm.dtype,
                    reduced[rank].storage_ref, reduced[rank].storage_bytes, chunk_bytes,
                )
                values.update((item.ref, item) for item in (peer_chunks[rank], outputs[rank]))
                rep_sends[rank] = _action(
                    rank=rank, kind=SwizzleActionKind.SEND,
                    stage=UnfusedComparisonStage.REPLICATION,
                    member_ref=coll.node_ref, deps=(reductions[rank].id,),
                    reads=(reduced[rank].ref,), peer_rank=peer,
                    route=_route(problem, rank, peer), logical_bytes=chunk_bytes,
                )
                actions_by_rank[rank].append(rep_sends[rank])
            for rank in (0, 1):
                peer = 1 - rank
                recv = _action(
                    rank=rank, kind=SwizzleActionKind.RECV,
                    stage=UnfusedComparisonStage.REPLICATION,
                    member_ref=coll.node_ref, deps=(rep_sends[peer].id,),
                    writes=(peer_chunks[rank].ref,), peer_rank=peer,
                    route=_route(problem, peer, rank), logical_bytes=chunk_bytes,
                )
                wait = _action(
                    rank=rank, kind=SwizzleActionKind.WAIT,
                    stage=UnfusedComparisonStage.REPLICATION,
                    member_ref=coll.node_ref, deps=(recv.id,), reads=(peer_chunks[rank].ref,),
                )
                complete = _action(
                    rank=rank, kind=SwizzleActionKind.BARRIER,
                    stage=UnfusedComparisonStage.COMPLETE,
                    member_ref=coll.node_ref,
                    deps=(reductions[rank].id, rep_sends[rank].id, wait.id),
                    reads=(outputs[rank].ref,),
                )
                actions_by_rank[rank].extend((recv, wait, complete))
    return tuple(tuple(items) for items in actions_by_rank), values


def build_unfused_comparison_plan(
    ir1: IR1,
    problem: SwizzleProblem,
    baseline: SwizzleCandidate,
) -> UnfusedComparisonPlan:
    ir1.validate("ir1")
    problem.validate("problem")
    baseline.validate("baseline")
    if baseline.algorithm is not SwizzleAlgorithm.UNFUSED or problem.source_ir1_id != ir1.id:
        raise SchemaError("requires exact IR1 problem and UNFUSED baseline", path="baseline")
    actions, _values = _build_actions(problem, baseline.semantic_witness)
    result = UnfusedComparisonPlan.create(
        source_ir1_id=ir1.id,
        problem=problem,
        baseline=baseline,
        pattern=problem.pattern,
        rank_programs=tuple(
            UnfusedComparisonRankProgram(rank, actions[index])
            for index, rank in enumerate(problem.collective.participant_ranks)
        ),
    )
    result.validate()
    return result


def project_unfused_comparison(
    ir1: IR1,
    plan: UnfusedComparisonPlan,
) -> UnfusedComparisonProjection:
    plan.validate_against(ir1)
    actions, values = _build_actions(plan.problem, plan.baseline.semantic_witness)
    if tuple(program.actions for program in plan.rank_programs) != actions:
        raise SchemaError("plan is not the exact deterministic baseline", path="plan.rank_programs")
    operands = []
    for rank_actions in actions:
        for action in rank_actions:
            refs = action.read_value_refs + action.write_value_refs
            uses = (SwizzleValueUse.READ,) * len(action.read_value_refs) + (SwizzleValueUse.WRITE,) * len(action.write_value_refs)
            for ordinal, (use, ref) in enumerate(zip(uses, refs, strict=True)):
                spec = values[ref]
                operands.append(UnfusedComparisonOperand(
                    action.id, ordinal, use, ref, spec.source_tensor_ref,
                    spec.shape,
                    spec.tensor_offset or (0,) * len(spec.shape),
                    spec.layout, spec.dtype, spec.bytes,
                    spec.storage_ref or spec.ref,
                    spec.storage_bytes or spec.bytes,
                    spec.byte_offset,
                ))
    flows = tuple(
        UnfusedComparisonFlow.create(
            route_ref=send.route_ref,
            source_rank=send.rank,
            destination_rank=recv.rank,
            die_path=send.die_path,
            send_task_ref=send.id,
            recv_task_ref=recv.id,
            logical_bytes=send.logical_bytes,
        )
        for rank_actions in actions
        for send in rank_actions
        if send.kind is SwizzleActionKind.SEND
        for recv in (
            next(
                item
                for other_actions in actions
                for item in other_actions
                if item.kind is SwizzleActionKind.RECV
                and item.stage is send.stage
                and item.rank == send.peer_rank
                and item.peer_rank == send.rank
            ),
        )
    )
    die_by_rank = {}
    for flow in flows:
        previous = die_by_rank.setdefault(flow.source_rank, flow.die_path[0])
        if previous != flow.die_path[0]:
            raise SchemaError("rank source die is inconsistent", path="problem.group.routes")
        previous = die_by_rank.setdefault(flow.destination_rank, flow.die_path[-1])
        if previous != flow.die_path[-1]:
            raise SchemaError("rank destination die is inconsistent", path="problem.group.routes")
    result = UnfusedComparisonProjection.create(
        source_ir1_id=ir1.id,
        source_plan_ref=plan.id,
        problem_ref=plan.problem.id,
        baseline_ref=plan.baseline.id,
        pattern=plan.pattern,
        ranks=tuple(
            UnfusedComparisonRankProjection(
                rank,
                die_by_rank[rank],
                tuple(action.id for action in actions[index]),
                actions[index][-1].id,
            )
            for index, rank in enumerate(plan.problem.collective.participant_ranks)
        ),
        operands=tuple(operands),
        flows=flows,
    )
    result.validate()
    return result


__all__ = ["build_unfused_comparison_plan", "project_unfused_comparison"]
