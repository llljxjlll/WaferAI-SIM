from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.external_memory import (
    SparseMemoryImage,
)
from llm.frontend.wafer_frontend.passes.offload import (
    execute_offload_blocking as _execute_offload_blocking,
    plan_offload_blocking as _plan_offload_blocking,
    plan_resident_only_memory,
    validate_blocking_offload_plan as _validate_blocking_offload_plan,
)
from llm.frontend.wafer_frontend.passes.memory_plan import (
    plan_hierarchical_memory,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
    ExternalTransferDirection,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryAllocationRequest,
    MemoryObjectKind,
    MemoryPlan,
    MemoryResidency,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
    ResidencyStatus,
)
from llm.frontend.wafer_frontend.schema.offload import (
    BlockingOffloadPlan,
    OffloadChunk,
    OffloadEventKind,
    OffloadOperation,
    OffloadOperationKind,
    OffloadResidencyRequirement,
    OffloadStateMapping,
    OffloadTraceEvent,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    from_data,
    to_primitive,
)


def fixture(
    *,
    hbm_bytes: int = 64,
    chunk_sizes: tuple[int, ...] = (64, 64),
) -> tuple[ExternalMemoryFabric, tuple[OffloadChunk, ...]]:
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL,
        location_ref="host:0",
        base_address=0,
        capacity_bytes=512,
        alignment_bytes=16,
    )
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref="die:0",
        base_address=1024,
        capacity_bytes=hbm_bytes,
        alignment_bytes=16,
    )
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=0,
        bytes_per_cycle=16,
        latency_cycles=3,
        queue_depth=1,
        max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id,
        hbm_capacity_ref=hbm.id,
        target_die_id=0,
        route_die_ids=(0,),
        route_latency_cycles=0,
        route_bytes_per_cycle=None,
    )
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,),
        hbm_capacities=(hbm,),
        links=(link,),
        connections=(connection,),
    )
    chunks = tuple(
        OffloadChunk.create(
            initial_version=MemoryStateVersion.create(
                state_ref=f"state:{index}",
                generation=0,
                predecessor_ref=None,
                writable=True,
            ),
            object_kind=MemoryObjectKind.PARAMETER,
            size_bytes=size,
            alignment_bytes=16,
            external_capacity_ref=external.id,
            external_address=sum(chunk_sizes[:index]),
            connection_ref=connection.id,
        )
        for index, size in enumerate(chunk_sizes)
    )
    return fabric, chunks


def event(
    ordinal: int,
    kind: OffloadEventKind,
    chunk: OffloadChunk,
    *,
    requirement: OffloadResidencyRequirement = (
        OffloadResidencyRequirement.AUTO_BRING_IN
    ),
) -> OffloadTraceEvent:
    return OffloadTraceEvent.create(
        ordinal=ordinal,
        kind=kind,
        chunk_ref=chunk.id,
        residency_requirement=requirement,
        deps=() if ordinal == 0 else (),
    )


def images_for(
    fabric: ExternalMemoryFabric,
) -> tuple[dict[str, SparseMemoryImage], dict[str, SparseMemoryImage]]:
    external = {
        item.id: SparseMemoryImage(item)
        for item in fabric.external_capacities
    }
    hbm = {
        item.id: SparseMemoryImage(item)
        for item in fabric.hbm_capacities
    }
    return external, hbm


def source_binding(
    fabric: ExternalMemoryFabric,
    chunks: tuple[OffloadChunk, ...],
    *,
    case: str = "case-a",
) -> dict[str, object]:
    external_by_id = {
        item.id: item for item in fabric.external_capacities
    }
    requests_by_chunk = {
        chunk.id: MemoryAllocationRequest.create(
            state_version_ref=chunk.initial_version.id,
            object_kind=chunk.object_kind,
            tier=MemoryTier.EXTERNAL,
            location_ref=external_by_id[
                chunk.external_capacity_ref
            ].location_ref,
            size_bytes=chunk.size_bytes,
            alignment_bytes=chunk.alignment_bytes,
            lifetime_start=0,
            lifetime_end_exclusive=1,
            pinned_address=chunk.external_address,
        )
        for chunk in chunks
    }
    source_plan = plan_hierarchical_memory(
        capacities=fabric.external_capacities,
        state_versions=tuple(chunk.initial_version for chunk in chunks),
        requests=tuple(requests_by_chunk.values()),
    )
    allocation_by_request = {
        item.request_ref: item for item in source_plan.allocations
    }
    mappings = tuple(
        OffloadStateMapping.create(
            chunk_ref=chunk.id,
            source_state_version_ref=chunk.initial_version.id,
            source_allocation_ref=allocation_by_request[
                requests_by_chunk[chunk.id].id
            ].id,
        )
        for chunk in chunks
    )
    return {
        "request_digest": canonical_digest((case, "request")),
        "logical_graph_digest": canonical_digest((case, "logical_graph")),
        "source_memory_plan_digest": canonical_digest(source_plan),
        "source_memory_plan": source_plan,
        "state_mappings": mappings,
    }


def plan_offload_blocking(
    *,
    fabric: ExternalMemoryFabric,
    chunks: tuple[OffloadChunk, ...],
    events: tuple[OffloadTraceEvent, ...],
) -> BlockingOffloadPlan:
    return _plan_offload_blocking(
        **source_binding(fabric, chunks),
        fabric=fabric,
        chunks=chunks,
        events=events,
    )


def validate_blocking_offload_plan(plan: BlockingOffloadPlan) -> None:
    _validate_blocking_offload_plan(
        plan,
        request_digest=plan.request_digest,
        logical_graph_digest=plan.logical_graph_digest,
        source_memory_plan_digest=plan.source_memory_plan_digest,
    )


def execute_offload_blocking(
    *,
    plan: BlockingOffloadPlan,
    external_images: dict[str, SparseMemoryImage],
    hbm_images: dict[str, SparseMemoryImage],
):
    return _execute_offload_blocking(
        plan=plan,
        request_digest=plan.request_digest,
        logical_graph_digest=plan.logical_graph_digest,
        source_memory_plan_digest=plan.source_memory_plan_digest,
        external_images=external_images,
        hbm_images=hbm_images,
    )


class BlockingOffloadTest(unittest.TestCase):
    def test_resident_only_oom_and_same_trace_offload_success(self) -> None:
        fabric, chunks = fixture()
        with self.assertRaisesRegex(SchemaError, "capacity exceeded") as caught:
            plan_resident_only_memory(fabric=fabric, chunks=chunks)
        self.assertEqual(caught.exception.code, "memory_capacity_exceeded")

        events = (
            event(0, OffloadEventKind.READ, chunks[0]),
            event(1, OffloadEventKind.READ, chunks[1]),
            event(2, OffloadEventKind.READ, chunks[0]),
        )
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=events,
        )
        self.assertEqual(plan.stats.consume_read_count, 3)
        self.assertEqual(plan.stats.bring_in_count, 3)
        self.assertEqual(plan.stats.clean_discard_count, 3)
        self.assertEqual(plan.stats.dirty_writeback_count, 0)
        self.assertLessEqual(
            plan.stats.hbm_peak_bytes,
            fabric.hbm_capacities[0].capacity_bytes,
        )
        self.assertEqual(plan.stats.final_resident_chunks, 0)
        self.assertEqual(plan.stats.final_pin_count, 0)
        self.assertFalse(plan.simulator_runtime_integrated)
        repeated = plan_offload_blocking(
            fabric=fabric,
            chunks=tuple(reversed(chunks)),
            events=events,
        )
        self.assertEqual(plan, repeated)
        decoded = from_data(
            BlockingOffloadPlan,
            to_primitive(plan),
            path="plan",
        )
        validate_blocking_offload_plan(decoded)
        self.assertEqual(canonical_digest(plan), canonical_digest(decoded))

    def test_dirty_eviction_writes_back_and_advances_generation(self) -> None:
        fabric, chunks = fixture()
        events = (
            event(0, OffloadEventKind.WRITE, chunks[0]),
            event(1, OffloadEventKind.READ, chunks[1]),
            event(2, OffloadEventKind.READ, chunks[0]),
        )
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=events,
        )
        self.assertEqual(plan.stats.consume_write_count, 1)
        self.assertEqual(plan.stats.dirty_writeback_count, 1)
        self.assertEqual(
            sum(
                item.direction
                is ExternalTransferDirection.HBM_TO_EXTERNAL
                for item in plan.transfer_requests
            ),
            1,
        )
        versions = sorted(
            (
                item
                for item in plan.state_versions
                if item.state_ref == chunks[0].initial_version.state_ref
            ),
            key=lambda item: item.generation,
        )
        self.assertEqual([item.generation for item in versions], [0, 1])
        self.assertEqual(versions[1].predecessor_ref, versions[0].id)
        write = next(
            item
            for item in plan.operations
            if item.kind is OffloadOperationKind.CONSUME_WRITE
        )
        self.assertEqual(write.state_version_ref, versions[1].id)
        self.assertTrue(write.dirty_after)
        generation_one_residency = next(
            item
            for item in plan.memory_plan.residencies
            if item.allocation_ref == write.hbm_allocation_ref
            and item.state_version_ref == versions[1].id
        )
        self.assertEqual(
            generation_one_residency.valid_from,
            write.ready_cycle,
        )
        self.assertEqual(
            generation_one_residency.status,
            ResidencyStatus.DIRTY,
        )

        allocation = next(
            item
            for item in plan.memory_plan.allocations
            if item.id == write.hbm_allocation_ref
        )
        allocation_request = next(
            item
            for item in plan.memory_plan.requests
            if item.id == allocation.request_ref
        )
        forged_residency = MemoryResidency.create(
            state_version_ref=versions[0].id,
            allocation_ref=allocation.id,
            status=ResidencyStatus.CLEAN,
            valid_from=allocation_request.lifetime_start,
            valid_until_exclusive=(
                allocation_request.lifetime_end_exclusive
            ),
        )
        forged_memory_plan = MemoryPlan.create(
            capacities=plan.memory_plan.capacities,
            state_versions=plan.memory_plan.state_versions,
            requests=plan.memory_plan.requests,
            allocations=plan.memory_plan.allocations,
            residencies=tuple(
                item
                for item in plan.memory_plan.residencies
                if item.allocation_ref != allocation.id
            )
            + (forged_residency,),
            peaks=plan.memory_plan.peaks,
        )
        forged_plan = BlockingOffloadPlan.create(
            request_digest=plan.request_digest,
            logical_graph_digest=plan.logical_graph_digest,
            source_memory_plan_digest=plan.source_memory_plan_digest,
            source_memory_plan=plan.source_memory_plan,
            state_mappings=plan.state_mappings,
            fabric=plan.fabric,
            chunks=plan.chunks,
            events=plan.events,
            state_versions=plan.state_versions,
            memory_plan=forged_memory_plan,
            transfer_requests=plan.transfer_requests,
            operations=plan.operations,
            stats=plan.stats,
        )
        with self.assertRaisesRegex(
            SchemaError,
            "no matching HBM residency",
        ):
            validate_blocking_offload_plan(forged_plan)

        external_images, hbm_images = images_for(fabric)
        external_capacity = fabric.external_capacities[0]
        external_images[external_capacity.id].write(0, b"A" * 128)
        execution = execute_offload_blocking(
            plan=plan,
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(
            execution.transfer_report.stats.completed_requests,
            len(plan.transfer_requests),
        )
        self.assertEqual(execution.transfer_report.stats.pending_requests, 0)
        self.assertFalse(execution.simulator_runtime_integrated)

    def test_next_use_evicts_farthest_future_chunk(self) -> None:
        fabric, chunks = fixture(
            hbm_bytes=128,
            chunk_sizes=(64, 64, 64),
        )
        events = (
            event(0, OffloadEventKind.READ, chunks[0]),
            event(1, OffloadEventKind.READ, chunks[1]),
            event(2, OffloadEventKind.READ, chunks[2]),
            event(3, OffloadEventKind.READ, chunks[0]),
            event(4, OffloadEventKind.READ, chunks[1]),
        )
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=events,
        )
        third_bring = [
            index
            for index, operation in enumerate(plan.operations)
            if operation.kind is OffloadOperationKind.BRING_IN
        ][2]
        victim = plan.operations[third_bring - 1]
        self.assertEqual(victim.kind, OffloadOperationKind.CLEAN_DISCARD)
        self.assertEqual(victim.chunk_ref, chunks[1].id)

        tied_plan = plan_offload_blocking(
            fabric=fabric,
            chunks=tuple(reversed(chunks)),
            events=(
                event(0, OffloadEventKind.READ, chunks[0]),
                event(1, OffloadEventKind.READ, chunks[1]),
                event(2, OffloadEventKind.READ, chunks[2]),
            ),
        )
        first_discard = next(
            item
            for item in tied_plan.operations
            if item.kind is OffloadOperationKind.CLEAN_DISCARD
        )
        self.assertEqual(
            first_discard.chunk_ref,
            min(chunks[0].id, chunks[1].id),
        )

    def test_read_before_resident_and_minimum_window_fail_closed(self) -> None:
        fabric, chunks = fixture()
        must_reside = (
            event(
                0,
                OffloadEventKind.READ,
                chunks[0],
                requirement=(
                    OffloadResidencyRequirement.MUST_ALREADY_RESIDENT
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "before chunk became") as caught:
            plan_offload_blocking(
                fabric=fabric,
                chunks=chunks,
                events=must_reside,
            )
        self.assertEqual(caught.exception.code, "offload_read_before_resident")

        tiny_fabric, large_chunk = fixture(
            hbm_bytes=64,
            chunk_sizes=(128,),
        )
        with self.assertRaisesRegex(SchemaError, "minimum HBM window") as caught:
            plan_offload_blocking(
                fabric=tiny_fabric,
                chunks=large_chunk,
                events=(
                    event(0, OffloadEventKind.READ, large_chunk[0]),
                ),
            )
        self.assertEqual(
            caught.exception.code,
            "offload_minimum_window_insufficient",
        )

    def test_pin_blocks_eviction_and_leak_is_rejected(self) -> None:
        fabric, chunks = fixture()
        with self.assertRaisesRegex(SchemaError, "pinned chunks") as caught:
            plan_offload_blocking(
                fabric=fabric,
                chunks=chunks,
                events=(
                    event(0, OffloadEventKind.PIN, chunks[0]),
                    event(1, OffloadEventKind.READ, chunks[1]),
                    event(2, OffloadEventKind.UNPIN, chunks[0]),
                ),
            )
        self.assertEqual(caught.exception.code, "offload_pinned_capacity")

        with self.assertRaisesRegex(SchemaError, "leaked") as caught:
            plan_offload_blocking(
                fabric=fabric,
                chunks=chunks,
                events=(
                    event(0, OffloadEventKind.PIN, chunks[0]),
                ),
            )
        self.assertEqual(caught.exception.code, "offload_pin_leak")

    def test_consumers_wait_for_bring_in_completion(self) -> None:
        fabric, chunks = fixture()
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=(event(0, OffloadEventKind.READ, chunks[0]),),
        )
        bring = next(
            item
            for item in plan.operations
            if item.kind is OffloadOperationKind.BRING_IN
        )
        consume = next(
            item
            for item in plan.operations
            if item.kind is OffloadOperationKind.CONSUME_READ
        )
        self.assertEqual(consume.start_cycle, bring.ready_cycle)
        self.assertEqual(consume.depends_on, (bring.id,))

    def test_hbm_reuse_before_dirty_transfer_completion_is_rejected(self) -> None:
        fabric, chunks = fixture()
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=(
                event(0, OffloadEventKind.WRITE, chunks[0]),
                event(1, OffloadEventKind.READ, chunks[1]),
            ),
        )
        writeback_index = next(
            index
            for index, operation in enumerate(plan.operations)
            if operation.kind is OffloadOperationKind.DIRTY_WRITEBACK
        )
        bring_index = writeback_index + 1
        original = plan.operations[bring_index]
        forged = OffloadOperation.create(
            sequence=original.sequence,
            kind=original.kind,
            chunk_ref=original.chunk_ref,
            trace_event_ref=original.trace_event_ref,
            state_version_ref=original.state_version_ref,
            hbm_allocation_ref=original.hbm_allocation_ref,
            transfer_request_ref=original.transfer_request_ref,
            depends_on=original.depends_on,
            start_cycle=plan.operations[writeback_index].ready_cycle - 1,
            ready_cycle=original.ready_cycle - 1,
            pin_count_after=original.pin_count_after,
            dirty_after=original.dirty_after,
        )
        operations = tuple(
            forged if index == bring_index else operation
            for index, operation in enumerate(plan.operations)
        )
        forged_plan = BlockingOffloadPlan.create(
            request_digest=plan.request_digest,
            logical_graph_digest=plan.logical_graph_digest,
            source_memory_plan_digest=plan.source_memory_plan_digest,
            source_memory_plan=plan.source_memory_plan,
            state_mappings=plan.state_mappings,
            fabric=plan.fabric,
            chunks=plan.chunks,
            events=plan.events,
            state_versions=plan.state_versions,
            memory_plan=plan.memory_plan,
            transfer_requests=plan.transfer_requests,
            operations=operations,
            stats=plan.stats,
        )
        with self.assertRaisesRegex(SchemaError, "transfer is in progress") as caught:
            validate_blocking_offload_plan(forged_plan)
        self.assertEqual(
            caught.exception.code,
            "offload_transfer_in_progress_reuse",
        )

    def test_cross_workload_digest_and_state_binding_fail_closed(self) -> None:
        fabric, chunks = fixture()
        plan = plan_offload_blocking(
            fabric=fabric,
            chunks=chunks,
            events=(event(0, OffloadEventKind.READ, chunks[0]),),
        )
        other = source_binding(fabric, chunks, case="case-b")
        wrong_digests = {
            "request_digest": other["request_digest"],
            "logical_graph_digest": other["logical_graph_digest"],
            "source_memory_plan_digest": canonical_digest(
                "another-source-plan"
            ),
        }
        for name, wrong_digest in wrong_digests.items():
            expected = {
                "request_digest": plan.request_digest,
                "logical_graph_digest": plan.logical_graph_digest,
                "source_memory_plan_digest": plan.source_memory_plan_digest,
            }
            expected[name] = wrong_digest
            with self.assertRaisesRegex(
                SchemaError,
                "different workload source",
            ) as caught:
                _validate_blocking_offload_plan(plan, **expected)
            self.assertEqual(
                caught.exception.code,
                "offload_workload_binding_mismatch",
            )

        foreign_version = MemoryStateVersion.create(
            state_ref="state:foreign",
            generation=0,
            predecessor_ref=None,
            writable=True,
        )
        foreign_chunk = OffloadChunk.create(
            initial_version=foreign_version,
            object_kind=chunks[0].object_kind,
            size_bytes=chunks[0].size_bytes,
            alignment_bytes=chunks[0].alignment_bytes,
            external_capacity_ref=chunks[0].external_capacity_ref,
            external_address=chunks[0].external_address,
            connection_ref=chunks[0].connection_ref,
        )
        original_binding = source_binding(fabric, (chunks[0],))
        with self.assertRaisesRegex(
            SchemaError,
            "unknown offload chunk|exactly cover",
        ) as caught:
            _plan_offload_blocking(
                **original_binding,
                fabric=fabric,
                chunks=(foreign_chunk,),
                events=(
                    event(0, OffloadEventKind.READ, foreign_chunk),
                ),
            )
        self.assertEqual(
            caught.exception.code,
            "offload_state_mapping_mismatch",
        )

        bad_source_digest = dict(original_binding)
        bad_source_digest["source_memory_plan_digest"] = canonical_digest(
            "another-source-plan"
        )
        with self.assertRaisesRegex(
            SchemaError,
            "does not match embedded source memory plan",
        ) as caught:
            _plan_offload_blocking(
                **bad_source_digest,
                fabric=fabric,
                chunks=(chunks[0],),
                events=(event(0, OffloadEventKind.READ, chunks[0]),),
            )
        self.assertEqual(
            caught.exception.code,
            "offload_source_digest_mismatch",
        )


if __name__ == "__main__":
    unittest.main()
