from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.external_memory import (
    SparseMemoryImage,
    execute_external_transfers,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
    ExternalTransferDirection,
    ExternalTransferReport,
    ExternalTransferRequest,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    from_data,
    to_primitive,
)


def memory_capacity(
    tier: MemoryTier,
    location: str,
    base: int,
    size: int = 256,
) -> MemoryTierCapacity:
    return MemoryTierCapacity.create(
        tier=tier,
        location_ref=location,
        base_address=base,
        capacity_bytes=size,
        alignment_bytes=16,
    )


def fabric(
    *,
    queue_depth: int = 2,
    max_outstanding: int = 3,
    include_remote: bool = True,
) -> ExternalMemoryFabric:
    external = memory_capacity(
        MemoryTier.EXTERNAL,
        "host:0",
        0,
    )
    hbm0 = memory_capacity(MemoryTier.HBM, "die:0", 1024)
    hbm = [hbm0]
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=0,
        bytes_per_cycle=16,
        latency_cycles=3,
        queue_depth=queue_depth,
        max_outstanding=max_outstanding,
    )
    connections = [
        ExternalMemoryConnection.create(
            link_ref=link.id,
            hbm_capacity_ref=hbm0.id,
            target_die_id=0,
            route_die_ids=(0,),
            route_latency_cycles=0,
            route_bytes_per_cycle=None,
        )
    ]
    if include_remote:
        hbm1 = memory_capacity(MemoryTier.HBM, "die:1", 2048)
        hbm.append(hbm1)
        connections.append(
            ExternalMemoryConnection.create(
                link_ref=link.id,
                hbm_capacity_ref=hbm1.id,
                target_die_id=1,
                route_die_ids=(0, 1),
                route_latency_cycles=5,
                route_bytes_per_cycle=8,
            )
        )
    return ExternalMemoryFabric.create(
        external_capacities=(external,),
        hbm_capacities=tuple(hbm),
        links=(link,),
        connections=tuple(connections),
    )


def images_for(
    value: ExternalMemoryFabric,
) -> tuple[dict[str, SparseMemoryImage], dict[str, SparseMemoryImage]]:
    return (
        {
            capacity.id: SparseMemoryImage(capacity)
            for capacity in value.external_capacities
        },
        {
            capacity.id: SparseMemoryImage(capacity)
            for capacity in value.hbm_capacities
        },
    )


def connection_for(
    value: ExternalMemoryFabric,
    die_id: int,
) -> ExternalMemoryConnection:
    return next(
        item for item in value.connections if item.target_die_id == die_id
    )


class ExternalMemoryTest(unittest.TestCase):
    def test_nonzero_payload_round_trip_and_directional_byte_stats(self) -> None:
        topology = fabric()
        external_images, hbm_images = images_for(topology)
        external = topology.external_capacities[0]
        hbm = next(
            item
            for item in topology.hbm_capacities
            if item.location_ref == "die:0"
        )
        payload = bytes((1, 7, 0, 9, 13, 0, 255, 2, 3))
        external_images[external.id].write(0, payload)
        direct = connection_for(topology, 0)
        load = ExternalTransferRequest.create(
            connection_ref=direct.id,
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=0,
            hbm_address=hbm.base_address,
            size_bytes=len(payload),
            issue_cycle=0,
        )
        store = ExternalTransferRequest.create(
            connection_ref=direct.id,
            direction=ExternalTransferDirection.HBM_TO_EXTERNAL,
            external_address=64,
            hbm_address=hbm.base_address,
            size_bytes=len(payload),
            issue_cycle=4,
        )
        report = execute_external_transfers(
            fabric=topology,
            requests=(store, load),
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(
            hbm_images[hbm.id].read(hbm.base_address, len(payload)),
            payload,
        )
        self.assertEqual(
            external_images[external.id].read(64, len(payload)),
            payload,
        )
        self.assertTrue(all(item.payload_digest != "0" * 64 for item in report.completions))
        self.assertEqual(report.stats.external_read_bytes, len(payload))
        self.assertEqual(report.stats.external_write_bytes, len(payload))
        self.assertEqual(report.stats.hbm_read_bytes, len(payload))
        self.assertEqual(report.stats.hbm_write_bytes, len(payload))
        self.assertEqual(report.stats.completed_requests, 2)
        self.assertEqual(report.stats.pending_requests, 0)
        self.assertFalse(report.simulator_runtime_integrated)

        decoded = from_data(
            ExternalTransferReport,
            to_primitive(report),
            path="report",
        )
        self.assertEqual(canonical_digest(decoded), canonical_digest(report))

    def test_direct_and_remote_latency_formulas_include_route_cost(self) -> None:
        topology = fabric()
        external_images, hbm_images = images_for(topology)
        external = topology.external_capacities[0]
        external_images[external.id].write(0, bytes(range(64)))
        direct = connection_for(topology, 0)
        remote = connection_for(topology, 1)
        hbm0 = next(
            item for item in topology.hbm_capacities if item.location_ref == "die:0"
        )
        hbm1 = next(
            item for item in topology.hbm_capacities if item.location_ref == "die:1"
        )
        direct_request = ExternalTransferRequest.create(
            connection_ref=direct.id,
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=0,
            hbm_address=hbm0.base_address,
            size_bytes=64,
            issue_cycle=0,
        )
        remote_request = ExternalTransferRequest.create(
            connection_ref=remote.id,
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=0,
            hbm_address=hbm1.base_address,
            size_bytes=64,
            issue_cycle=7,
        )
        report = execute_external_transfers(
            fabric=topology,
            requests=(remote_request, direct_request),
            external_images=external_images,
            hbm_images=hbm_images,
        )
        direct_completion, remote_completion = report.completions
        self.assertEqual(direct_completion.external_service_cycles, 3 + 4)
        self.assertEqual(direct_completion.route_service_cycles, 0)
        self.assertEqual(direct_completion.completion_cycle, 7)
        self.assertEqual(remote_completion.external_service_cycles, 3 + 4)
        self.assertEqual(remote_completion.route_service_cycles, 5 + 8)
        self.assertEqual(remote_completion.completion_cycle, 27)
        self.assertEqual(
            hbm_images[hbm1.id].read(hbm1.base_address, 64),
            bytes(range(64)),
        )

    def test_opposite_directions_share_one_half_duplex_service(self) -> None:
        topology = fabric(include_remote=False)
        external_images, hbm_images = images_for(topology)
        external = topology.external_capacities[0]
        hbm = topology.hbm_capacities[0]
        external_images[external.id].write(0, b"A" * 64)
        hbm_images[hbm.id].write(hbm.base_address + 64, b"B" * 64)
        direct = topology.connections[0]
        requests = (
            ExternalTransferRequest.create(
                connection_ref=direct.id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=0,
                hbm_address=hbm.base_address,
                size_bytes=64,
                issue_cycle=0,
            ),
            ExternalTransferRequest.create(
                connection_ref=direct.id,
                direction=ExternalTransferDirection.HBM_TO_EXTERNAL,
                external_address=128,
                hbm_address=hbm.base_address + 64,
                size_bytes=64,
                issue_cycle=0,
            ),
        )
        report = execute_external_transfers(
            fabric=topology,
            requests=requests,
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(
            tuple(item.start_cycle for item in report.completions),
            (0, 7),
        )
        self.assertEqual(
            tuple(item.completion_cycle for item in report.completions),
            (7, 14),
        )
        self.assertEqual(report.stats.shared_link_busy_cycles, 14)
        self.assertEqual(report.stats.queue_stall_cycles, 7)
        self.assertEqual(report.stats.max_queue_occupancy, 1)
        self.assertEqual(report.stats.max_outstanding_observed, 2)

    def test_cross_die_requests_contend_but_distinct_links_run_in_parallel(self) -> None:
        shared = fabric()
        external_images, hbm_images = images_for(shared)
        external = shared.external_capacities[0]
        external_images[external.id].write(0, b"C" * 128)
        shared_requests = tuple(
            ExternalTransferRequest.create(
                connection_ref=connection_for(shared, die_id).id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=die_id * 64,
                hbm_address=next(
                    item.base_address
                    for item in shared.hbm_capacities
                    if item.location_ref == f"die:{die_id}"
                ),
                size_bytes=64,
                issue_cycle=0,
            )
            for die_id in (0, 1)
        )
        shared_report = execute_external_transfers(
            fabric=shared,
            requests=shared_requests,
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(shared_report.completions[0].start_cycle, 0)
        self.assertEqual(
            shared_report.completions[1].start_cycle,
            shared_report.completions[0].completion_cycle,
        )

        external0 = memory_capacity(MemoryTier.EXTERNAL, "host:0", 0)
        external1 = memory_capacity(MemoryTier.EXTERNAL, "host:1", 512)
        hbm0 = memory_capacity(MemoryTier.HBM, "die:0", 1024)
        hbm1 = memory_capacity(MemoryTier.HBM, "die:1", 2048)
        links = tuple(
            ExternalMemoryLink.create(
                external_capacity_ref=capacity.id,
                ingress_die_id=index,
                bytes_per_cycle=16,
                latency_cycles=3,
                queue_depth=1,
                max_outstanding=2,
            )
            for index, capacity in enumerate((external0, external1))
        )
        connections = tuple(
            ExternalMemoryConnection.create(
                link_ref=links[index].id,
                hbm_capacity_ref=hbm.id,
                target_die_id=index,
                route_die_ids=(index,),
                route_latency_cycles=0,
                route_bytes_per_cycle=None,
            )
            for index, hbm in enumerate((hbm0, hbm1))
        )
        independent = ExternalMemoryFabric.create(
            external_capacities=(external0, external1),
            hbm_capacities=(hbm0, hbm1),
            links=links,
            connections=connections,
        )
        external_images, hbm_images = images_for(independent)
        for capacity in independent.external_capacities:
            external_images[capacity.id].write(
                capacity.base_address,
                b"P" * 64,
            )
        independent_requests = tuple(
            ExternalTransferRequest.create(
                connection_ref=connections[index].id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=external_capacity.base_address,
                hbm_address=hbm_capacity.base_address,
                size_bytes=64,
                issue_cycle=0,
            )
            for index, (external_capacity, hbm_capacity) in enumerate(
                ((external0, hbm0), (external1, hbm1))
            )
        )
        independent_report = execute_external_transfers(
            fabric=independent,
            requests=independent_requests,
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(
            tuple(item.start_cycle for item in independent_report.completions),
            (0, 0),
        )
        self.assertEqual(
            tuple(item.completion_cycle for item in independent_report.completions),
            (7, 7),
        )
        self.assertEqual(independent_report.stats.makespan_cycles, 7)

    def test_queue_and_outstanding_exhaustion_fail_closed(self) -> None:
        topology = fabric(
            queue_depth=1,
            max_outstanding=2,
            include_remote=False,
        )
        external_images, hbm_images = images_for(topology)
        external = topology.external_capacities[0]
        hbm = topology.hbm_capacities[0]
        external_images[external.id].write(0, b"X" * 192)
        direct = topology.connections[0]
        requests = tuple(
            ExternalTransferRequest.create(
                connection_ref=direct.id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=index * 64,
                hbm_address=hbm.base_address + index * 64,
                size_bytes=64,
                issue_cycle=0,
            )
            for index in range(3)
        )
        with self.assertRaisesRegex(SchemaError, "exhausted") as caught:
            execute_external_transfers(
                fabric=topology,
                requests=requests,
                external_images=external_images,
                hbm_images=hbm_images,
            )
        self.assertEqual(caught.exception.code, "external_queue_exhausted")

    def test_missing_connection_and_out_of_range_address_fail_closed(self) -> None:
        topology = fabric(include_remote=False)
        external_images, hbm_images = images_for(topology)
        hbm = topology.hbm_capacities[0]
        missing = ExternalTransferRequest.create(
            connection_ref="missing_connection",
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=0,
            hbm_address=hbm.base_address,
            size_bytes=16,
            issue_cycle=0,
        )
        with self.assertRaisesRegex(SchemaError, "no declared") as caught:
            execute_external_transfers(
                fabric=topology,
                requests=(missing,),
                external_images=external_images,
                hbm_images=hbm_images,
            )
        self.assertEqual(caught.exception.code, "external_connection_missing")

        direct = topology.connections[0]
        out_of_range = ExternalTransferRequest.create(
            connection_ref=direct.id,
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=250,
            hbm_address=hbm.base_address,
            size_bytes=16,
            issue_cycle=0,
        )
        with self.assertRaisesRegex(SchemaError, "exceeds configured") as caught:
            execute_external_transfers(
                fabric=topology,
                requests=(out_of_range,),
                external_images=external_images,
                hbm_images=hbm_images,
            )
        self.assertEqual(
            caught.exception.code,
            "external_memory_address_out_of_range",
        )

    def test_invalid_link_route_capacity_and_direction_are_rejected(self) -> None:
        with self.assertRaisesRegex(SchemaError, "greater than zero"):
            memory_capacity(MemoryTier.EXTERNAL, "host:0", 0, size=0)
        topology = fabric()
        link = topology.links[0]
        with self.assertRaisesRegex(SchemaError, "queue_depth"):
            ExternalMemoryLink.create(
                external_capacity_ref=link.external_capacity_ref,
                ingress_die_id=0,
                bytes_per_cycle=16,
                latency_cycles=3,
                queue_depth=1,
                max_outstanding=3,
            )

        remote = connection_for(topology, 1)
        bad_remote = ExternalMemoryConnection.create(
            link_ref=remote.link_ref,
            hbm_capacity_ref=remote.hbm_capacity_ref,
            target_die_id=1,
            route_die_ids=(0, 1),
            route_latency_cycles=0,
            route_bytes_per_cycle=None,
        )
        with self.assertRaisesRegex(SchemaError, "explicit nonzero route"):
            ExternalMemoryFabric.create(
                external_capacities=topology.external_capacities,
                hbm_capacities=topology.hbm_capacities,
                links=topology.links,
                connections=tuple(
                    bad_remote if item.id == remote.id else item
                    for item in topology.connections
                ),
            )

        raw = to_primitive(
            ExternalTransferRequest.create(
                connection_ref=topology.connections[0].id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=0,
                hbm_address=1024,
                size_bytes=16,
                issue_cycle=0,
            )
        )
        assert isinstance(raw, dict)
        raw["direction"] = "hbm_to_sram"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            from_data(ExternalTransferRequest, raw, path="request")

    def test_overlapping_address_spaces_and_wrong_images_are_rejected(self) -> None:
        topology = fabric()
        first = next(
            item
            for item in topology.hbm_capacities
            if item.location_ref == "die:0"
        )
        overlapping = memory_capacity(
            MemoryTier.HBM,
            "die:0",
            first.base_address + 16,
            size=256,
        )
        remote = connection_for(topology, 1)
        replacement = ExternalMemoryConnection.create(
            link_ref=remote.link_ref,
            hbm_capacity_ref=overlapping.id,
            target_die_id=1,
            route_die_ids=remote.route_die_ids,
            route_latency_cycles=remote.route_latency_cycles,
            route_bytes_per_cycle=remote.route_bytes_per_cycle,
        )
        with self.assertRaisesRegex(SchemaError, "must not overlap"):
            ExternalMemoryFabric.create(
                external_capacities=topology.external_capacities,
                hbm_capacities=(first, overlapping),
                links=topology.links,
                connections=tuple(
                    replacement if item.id == remote.id else item
                    for item in topology.connections
                ),
            )

        external_images, hbm_images = images_for(topology)
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            execute_external_transfers(
                fabric=topology,
                requests=(),
                external_images={},
                hbm_images=hbm_images,
            )

    def test_hbm_local_addresses_are_scoped_by_owner_die(self) -> None:
        external = memory_capacity(MemoryTier.EXTERNAL, "host:0", 0)
        hbm0 = memory_capacity(MemoryTier.HBM, "die:0", 0)
        hbm1 = memory_capacity(MemoryTier.HBM, "die:1", 0)
        link = ExternalMemoryLink.create(
            external_capacity_ref=external.id,
            ingress_die_id=0,
            bytes_per_cycle=16,
            latency_cycles=3,
            queue_depth=2,
            max_outstanding=3,
        )
        direct = ExternalMemoryConnection.create(
            link_ref=link.id,
            hbm_capacity_ref=hbm0.id,
            target_die_id=0,
            route_die_ids=(0,),
            route_latency_cycles=0,
            route_bytes_per_cycle=None,
        )
        remote = ExternalMemoryConnection.create(
            link_ref=link.id,
            hbm_capacity_ref=hbm1.id,
            target_die_id=1,
            route_die_ids=(0, 1),
            route_latency_cycles=1,
            route_bytes_per_cycle=16,
        )
        topology = ExternalMemoryFabric.create(
            external_capacities=(external,),
            hbm_capacities=(hbm0, hbm1),
            links=(link,),
            connections=(direct, remote),
        )
        self.assertEqual(
            {item.base_address for item in topology.hbm_capacities},
            {0},
        )
        external_images, hbm_images = images_for(topology)
        external_images[external.id].write(0, b"A" * 16)
        external_images[external.id].write(64, b"B" * 16)
        requests = (
            ExternalTransferRequest.create(
                connection_ref=direct.id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=0,
                hbm_address=0,
                size_bytes=16,
                issue_cycle=0,
            ),
            ExternalTransferRequest.create(
                connection_ref=remote.id,
                direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
                external_address=64,
                hbm_address=0,
                size_bytes=16,
                issue_cycle=0,
            ),
        )
        execute_external_transfers(
            fabric=topology,
            requests=requests,
            external_images=external_images,
            hbm_images=hbm_images,
        )
        self.assertEqual(hbm_images[hbm0.id].read(0, 16), b"A" * 16)
        self.assertEqual(hbm_images[hbm1.id].read(0, 16), b"B" * 16)

    def test_connection_target_must_match_hbm_owner_die(self) -> None:
        external = memory_capacity(MemoryTier.EXTERNAL, "host:0", 0)
        hbm = memory_capacity(MemoryTier.HBM, "die:1", 0)
        link = ExternalMemoryLink.create(
            external_capacity_ref=external.id,
            ingress_die_id=0,
            bytes_per_cycle=16,
            latency_cycles=3,
            queue_depth=2,
            max_outstanding=3,
        )
        wrong = ExternalMemoryConnection.create(
            link_ref=link.id,
            hbm_capacity_ref=hbm.id,
            target_die_id=0,
            route_die_ids=(0,),
            route_latency_cycles=0,
            route_bytes_per_cycle=None,
        )
        with self.assertRaisesRegex(SchemaError, "does not own"):
            ExternalMemoryFabric.create(
                external_capacities=(external,),
                hbm_capacities=(hbm,),
                links=(link,),
                connections=(wrong,),
            )

    def test_forged_completion_timing_is_rejected(self) -> None:
        topology = fabric(include_remote=False)
        external_images, hbm_images = images_for(topology)
        external = topology.external_capacities[0]
        hbm = topology.hbm_capacities[0]
        external_images[external.id].write(0, b"Z" * 16)
        request = ExternalTransferRequest.create(
            connection_ref=topology.connections[0].id,
            direction=ExternalTransferDirection.EXTERNAL_TO_HBM,
            external_address=0,
            hbm_address=hbm.base_address,
            size_bytes=16,
            issue_cycle=0,
        )
        report = execute_external_transfers(
            fabric=topology,
            requests=(request,),
            external_images=external_images,
            hbm_images=hbm_images,
        )
        forged = replace(
            report.completions[0],
            completion_cycle=report.completions[0].completion_cycle - 1,
        )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(report, completions=(forged,)).validate()


if __name__ == "__main__":
    unittest.main()
