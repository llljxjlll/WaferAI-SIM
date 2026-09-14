from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import (
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes import (
    build_parallel_transport_plan,
)
from llm.frontend.wafer_frontend.schema import (
    ParallelCommunicationKind,
    ParallelCommunicationRequest,
    ParallelGroupKind,
    ParallelTransferEndpointRole,
    ParallelTransportPlan,
    RectMeshSpec,
    build_dense_parallel_placement,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware


def _fabric(
    columns: int,
    rows: int,
    *,
    noc_columns: int = 2,
    noc_rows: int = 2,
):
    hardware = minimal_hardware(columns, rows)
    hardware["x"] = noc_columns
    hardware["y"] = noc_rows
    hardware["cores"] = [
        {"id": core_id} for core_id in range(noc_columns * noc_rows)
    ]
    return physical_fabric_from_data(hardware)


def _collective(request_id, kind, group_id, transfer_bytes=64):
    return ParallelCommunicationRequest(
        id=request_id,
        workload_case_id="test_case",
        workload_request_digest="0" * 64,
        logical_operation_ref=f"op_{request_id}",
        workload_phase="test",
        workload_step=7,
        workload_layer=1,
        payload_value_refs=(f"value_{request_id}",),
        is_noop=transfer_bytes == 0,
        kind=kind,
        group_id=group_id,
        source_rank=None,
        destination_rank=None,
        transfer_bytes=transfer_bytes,
    )


def _p2p(request_id, source, destination, transfer_bytes=64):
    return ParallelCommunicationRequest(
        id=request_id,
        workload_case_id="test_case",
        workload_request_digest="0" * 64,
        logical_operation_ref=f"op_{request_id}",
        workload_phase="test",
        workload_step=7,
        workload_layer=1,
        payload_value_refs=(f"value_{request_id}",),
        is_noop=transfer_bytes == 0,
        kind=ParallelCommunicationKind.P2P,
        group_id=None,
        source_rank=source,
        destination_rank=destination,
        transfer_bytes=transfer_bytes,
    )


class ParallelTransportRouteTest(unittest.TestCase):
    def test_non_contiguous_dies_route_through_idle_dies(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 4),
            tp_degree=2,
            dp_degree=1,
            active_die_ids=(0, 3),
        )
        plan = build_parallel_transport_plan(
            placement,
            _fabric(4, 1),
            (_p2p("far", 0, 1, 96),),
        )

        self.assertEqual(placement.idle_die_ids, (1, 2))
        self.assertEqual(len(plan.routes), 1)
        self.assertEqual(plan.routes[0].die_path, (0, 1, 2, 3))
        self.assertEqual(
            tuple(hop.source_die for hop in plan.routes[0].hops),
            (0, 1, 2),
        )
        self.assertTrue(
            set(plan.routes[0].die_path[1:-1]).issubset(placement.idle_die_ids)
        )
        plan.validate_against(placement, _fabric(4, 1))

    def test_x_first_route_uses_physical_mesh_coordinates(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(3, 3),
            tp_degree=2,
            dp_degree=1,
            active_die_ids=(0, 8),
        )
        plan = build_parallel_transport_plan(
            placement,
            _fabric(3, 3),
            (_p2p("diagonal", 0, 1),),
        )

        self.assertEqual(plan.routes[0].die_path, (0, 1, 2, 5, 8))

    def test_single_member_collective_degenerates_to_local_noop(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 2),
            tp_degree=1,
            dp_degree=1,
        )
        tp_group = placement.select_groups(ParallelGroupKind.TP)[0]
        plan = build_parallel_transport_plan(
            placement,
            _fabric(2, 1),
            (
                _collective(
                    "singleton",
                    ParallelCommunicationKind.ALL_REDUCE,
                    tp_group.id,
                ),
            ),
        )

        self.assertEqual(plan.transfers, ())
        self.assertEqual(plan.routes, ())
        self.assertEqual(plan.endpoints, ())
        self.assertEqual(len(plan.core_bindings), 1)


class ParallelCollectiveLoweringTest(unittest.TestCase):
    def test_all_collective_kinds_have_exact_transfer_counts_and_pairs(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 3),
            tp_degree=3,
            dp_degree=1,
        )
        tp_group = placement.select_groups(ParallelGroupKind.TP)[0]
        requests = (
            _collective(
                "ag", ParallelCommunicationKind.ALL_GATHER, tp_group.id, 11
            ),
            _collective(
                "ar", ParallelCommunicationKind.ALL_REDUCE, tp_group.id, 13
            ),
            _collective(
                "a2a", ParallelCommunicationKind.ALL_TO_ALL, tp_group.id, 17
            ),
            _collective(
                "rs", ParallelCommunicationKind.REDUCE_SCATTER, tp_group.id, 19
            ),
        )
        plan = build_parallel_transport_plan(
            placement,
            _fabric(3, 1),
            requests,
        )

        counts = {
            request.id: sum(
                transfer.request_id == request.id for transfer in plan.transfers
            )
            for request in requests
        }
        self.assertEqual(counts, {"ag": 6, "ar": 12, "a2a": 6, "rs": 6})
        self.assertEqual(len(plan.routes), 6)
        self.assertEqual(len(plan.endpoints), 2 * len(plan.transfers))

        endpoints = {endpoint.id: endpoint for endpoint in plan.endpoints}
        requests_by_id = {request.id: request for request in requests}
        for transfer in plan.transfers:
            request = requests_by_id[transfer.request_id]
            self.assertEqual(transfer.workload_phase, request.workload_phase)
            self.assertEqual(transfer.workload_step, 7)
            self.assertEqual(transfer.workload_layer, 1)
            self.assertEqual(
                transfer.logical_operation_ref,
                request.logical_operation_ref,
            )
            self.assertEqual(
                transfer.payload_value_refs,
                request.payload_value_refs,
            )
            self.assertIn(
                transfer.algorithm_phase,
                {kind.value for kind in ParallelCommunicationKind},
            )
            self.assertLess(transfer.algorithm_step, 2)
            send = endpoints[transfer.send_endpoint_id]
            recv = endpoints[transfer.recv_endpoint_id]
            self.assertIs(send.role, ParallelTransferEndpointRole.SEND)
            self.assertIs(recv.role, ParallelTransferEndpointRole.RECV)
            self.assertEqual(
                (send.rank, send.peer_rank, send.bytes),
                (
                    transfer.source_rank,
                    transfer.destination_rank,
                    transfer.bytes,
                ),
            )
            self.assertEqual(
                (recv.rank, recv.peer_rank, recv.bytes),
                (
                    transfer.destination_rank,
                    transfer.source_rank,
                    transfer.bytes,
                ),
            )

    def test_explicit_zero_payload_is_a_transport_noop(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 2),
            tp_degree=2,
            dp_degree=1,
        )
        tp_group = placement.select_groups(ParallelGroupKind.TP)[0]
        request = _collective(
            "zero",
            ParallelCommunicationKind.ALL_TO_ALL,
            tp_group.id,
            0,
        )
        plan = build_parallel_transport_plan(
            placement,
            _fabric(2, 1),
            (request,),
        )

        self.assertTrue(plan.requests[0].is_noop)
        self.assertEqual(plan.transfers, ())
        self.assertEqual(plan.routes, ())
        self.assertEqual(plan.endpoints, ())

    def test_plan_round_trip_and_input_order_are_deterministic(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(2, 2),
            tp_degree=2,
            dp_degree=2,
        )
        group = placement.select_groups(ParallelGroupKind.DP)[0]
        requests = (
            _p2p("z-last", 0, 3, 9),
            _collective(
                "a-first",
                ParallelCommunicationKind.ALL_GATHER,
                group.id,
                7,
            ),
        )
        fabric = _fabric(2, 2)
        first = build_parallel_transport_plan(placement, fabric, requests)
        second = build_parallel_transport_plan(
            placement, fabric, tuple(reversed(requests))
        )

        self.assertEqual(first, second)
        decoded = loads_dataclass(ParallelTransportPlan, canonical_json(first))
        self.assertEqual(decoded, first)
        decoded.validate_against(placement, fabric)


class ParallelCoreBindingTest(unittest.TestCase):
    def test_bindings_follow_one_and_four_core_fabrics_without_fixed_stride(self) -> None:
        one_core_placement = build_dense_parallel_placement(
            RectMeshSpec(1, 4),
            tp_degree=2,
            dp_degree=1,
            active_die_ids=(0, 3),
        )
        one_core = build_parallel_transport_plan(
            one_core_placement,
            _fabric(4, 1, noc_columns=1, noc_rows=1),
            (),
        )
        self.assertEqual(
            tuple(binding.runtime_core_id for binding in one_core.core_bindings),
            (0, 3),
        )

        four_core_placement = build_dense_parallel_placement(
            RectMeshSpec(1, 4),
            tp_degree=2,
            dp_degree=1,
            active_die_ids=(0, 3),
            local_core_ids=(3, 3),
        )
        four_core = build_parallel_transport_plan(
            four_core_placement,
            _fabric(4, 1, noc_columns=2, noc_rows=2),
            (),
        )
        self.assertEqual(
            tuple(binding.runtime_core_id for binding in four_core.core_bindings),
            (3, 15),
        )
        self.assertEqual(
            tuple(binding.core_spec_ref for binding in four_core.core_bindings),
            ("core_d0_c3", "core_d3_c3"),
        )

    def test_out_of_range_local_core_fails_before_plan_creation(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 2),
            tp_degree=2,
            dp_degree=1,
            local_core_ids=(0, 4),
        )
        with self.assertRaisesRegex(SchemaError, "outside this Die"):
            build_parallel_transport_plan(
                placement,
                _fabric(2, 1, noc_columns=2, noc_rows=2),
                (),
            )


class ParallelTransportValidationTest(unittest.TestCase):
    def test_invalid_p2p_group_and_endpoint_mutation_fail_closed(self) -> None:
        placement = build_dense_parallel_placement(
            RectMeshSpec(1, 2), tp_degree=2, dp_degree=1
        )
        fabric = _fabric(2, 1)
        with self.assertRaises(SchemaError):
            build_parallel_transport_plan(
                placement, fabric, (_p2p("outside", 0, 2),)
            )
        with self.assertRaises(SchemaError):
            build_parallel_transport_plan(
                placement,
                fabric,
                (
                    _collective(
                        "missing",
                        ParallelCommunicationKind.ALL_GATHER,
                        "unknown_group",
                    ),
                ),
            )

        plan = build_parallel_transport_plan(
            placement, fabric, (_p2p("valid", 0, 1, 33),)
        )
        bad_endpoint = replace(plan.endpoints[0], bytes=34)
        mutation = replace(
            plan,
            endpoints=tuple(
                sorted(
                    (bad_endpoint, *plan.endpoints[1:]),
                    key=lambda item: item.id,
                )
            ),
        )
        with self.assertRaisesRegex(SchemaError, "endpoint"):
            mutation.validate()


if __name__ == "__main__":
    unittest.main()
