from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.external_dma_program import (
    finalize_external_dma_program,
    parse_external_dma_program,
    serialize_external_dma_program,
)
from llm.frontend.wafer_frontend.passes.memory_plan import (
    plan_hierarchical_memory,
)
from llm.frontend.wafer_frontend.passes.offload import (
    plan_offload_blocking,
)
from llm.frontend.wafer_frontend.schema.external_dma_program import (
    ExternalDmaBackendBinding,
    ExternalDmaProbe,
    ExternalDmaProgram,
    ExternalDmaSeed,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryAllocationRequest,
    MemoryObjectKind,
    MemoryStateVersion,
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.offload import (
    OffloadChunk,
    OffloadEventKind,
    OffloadStateMapping,
    OffloadTraceEvent,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    from_data,
    to_primitive,
)


PAYLOAD = bytes((1, 7, 0, 9, 13, 0, 255, 2, 8, 6, 7, 5, 3, 0, 9, 4))


def make_program() -> ExternalDmaProgram:
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL,
        location_ref="host:0",
        base_address=0,
        capacity_bytes=64,
        alignment_bytes=16,
    )
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref="die:0",
        base_address=1024,
        capacity_bytes=16,
        alignment_bytes=16,
    )
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=0,
        bytes_per_cycle=4,
        latency_cycles=2,
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
    version = MemoryStateVersion.create(
        state_ref="state:golden",
        generation=0,
        predecessor_ref=None,
        writable=True,
    )
    chunk = OffloadChunk.create(
        initial_version=version,
        object_kind=MemoryObjectKind.PARAMETER,
        size_bytes=len(PAYLOAD),
        alignment_bytes=16,
        external_capacity_ref=external.id,
        external_address=0,
        connection_ref=connection.id,
    )
    source_request = MemoryAllocationRequest.create(
        state_version_ref=version.id,
        object_kind=MemoryObjectKind.PARAMETER,
        tier=MemoryTier.EXTERNAL,
        location_ref=external.location_ref,
        size_bytes=len(PAYLOAD),
        alignment_bytes=16,
        lifetime_start=0,
        lifetime_end_exclusive=1,
        pinned_address=0,
    )
    source_plan = plan_hierarchical_memory(
        capacities=(external,),
        state_versions=(version,),
        requests=(source_request,),
    )
    source_allocation = source_plan.allocations[0]
    mapping = OffloadStateMapping.create(
        chunk_ref=chunk.id,
        source_state_version_ref=version.id,
        source_allocation_ref=source_allocation.id,
    )
    event = OffloadTraceEvent.create(
        ordinal=0,
        kind=OffloadEventKind.WRITE,
        chunk_ref=chunk.id,
    )
    plan = plan_offload_blocking(
        request_digest=canonical_digest(("golden", "request")),
        logical_graph_digest=canonical_digest(("golden", "graph")),
        source_memory_plan_digest=canonical_digest(source_plan),
        source_memory_plan=source_plan,
        state_mappings=(mapping,),
        fabric=fabric,
        chunks=(chunk,),
        events=(event,),
    )
    return finalize_external_dma_program(
        plan=plan,
        case_digest=canonical_digest(("golden", "case")),
        backend_bindings=(
            ExternalDmaBackendBinding.create(
                hbm_capacity_ref=hbm.id,
                owner_die_id=0,
                stack_id=0,
                channel_id=0,
            ),
        ),
        external_seeds=(
            ExternalDmaSeed.create(
                external_capacity_ref=external.id,
                address=0,
                payload=PAYLOAD,
            ),
        ),
        external_probes=(
            ExternalDmaProbe.create(
                external_capacity_ref=external.id,
                address=0,
                expected_payload=PAYLOAD,
            ),
        ),
    )


class ExternalDmaProgramTest(unittest.TestCase):
    def test_finalizer_is_stable_and_matches_cross_language_golden(self) -> None:
        program = make_program()
        self.assertEqual(len(program.descriptors), 2)
        self.assertEqual(program.descriptors[0].depends_on, ())
        self.assertEqual(
            program.descriptors[1].depends_on,
            (program.descriptors[0].id,),
        )
        payload = serialize_external_dma_program(program)
        self.assertEqual(parse_external_dma_program(payload), program)
        golden = (
            Path(__file__).parents[1]
            / "golden"
            / "external_dma_program_v1alpha1.json"
        )
        self.assertEqual(golden.read_text(encoding="utf-8"), payload)
        self.assertEqual(make_program(), program)

    def test_source_digest_and_binding_mismatch_fail_closed(self) -> None:
        program = make_program()
        with self.assertRaisesRegex(SchemaError, "SHA-256"):
            replace(program, case_digest="not-a-digest").validate()
        binding = program.backend_bindings[0]
        with self.assertRaisesRegex(SchemaError, "owner differs"):
            replace(
                program,
                backend_bindings=(
                    ExternalDmaBackendBinding.create(
                        hbm_capacity_ref=binding.hbm_capacity_ref,
                        owner_die_id=1,
                        stack_id=binding.stack_id,
                        channel_id=binding.channel_id,
                    ),
                ),
            ).validate()

    def test_dependency_and_address_tampering_fail_closed(self) -> None:
        primitive = to_primitive(make_program())
        assert isinstance(primitive, dict)
        descriptors = primitive["descriptors"]
        assert isinstance(descriptors, list)
        second = descriptors[1]
        assert isinstance(second, dict)
        second["depends_on"] = []
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            from_data(
                ExternalDmaProgram,
                primitive,
                path="external_dma_program",
            )

        program = make_program()
        first = program.descriptors[0]
        descriptor = type(first).create(
            sequence=first.sequence,
            operation_ref=first.operation_ref,
            source_transfer_request_ref=(
                first.source_transfer_request_ref
            ),
            source_operation_deps=first.source_operation_deps,
            depends_on=first.depends_on,
            connection_ref=first.connection_ref,
            direction=first.direction,
            external_address=first.external_address,
            hbm_address=first.hbm_address + 16,
            size_bytes=first.size_bytes,
            planned_issue_cycle=first.planned_issue_cycle,
            planned_ready_cycle=first.planned_ready_cycle,
        )
        with self.assertRaisesRegex(SchemaError, "exceeds HBM"):
            replace(
                program,
                descriptors=(descriptor, program.descriptors[1]),
            ).validate()


if __name__ == "__main__":
    unittest.main()
