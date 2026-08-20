from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.stage4_state_transfer import (
    build_stage4_state_transfers,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    KvTransferSegment,
    SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
)

from test_stage4_carriers import _chain


def _case(prefill_tp: int, decode_tp: int):
    planned = _chain(prefill_tp, decode_tp)[-1]
    logical = build_stage4_state_transfers(
        planned.graph,
        planned.pd_plan,
    )
    contracts = tuple(
        SegmentedKvStateTransferContract.from_logical_slice(
            producer_pass="stage4_segmented_state_transfer_test",
            logical_slice=item,
        )
        for item in logical
    )
    for contract in contracts:
        contract.validate_against(planned.graph)
    return planned, logical, contracts


class Stage4SegmentedStateTransferSchemaTest(unittest.TestCase):
    def test_gather_and_scatter_segments_are_exact_and_stable(self) -> None:
        self.assertEqual(
            SEGMENTED_KV_STATE_TRANSFER_CONTRACT_SCHEMA_VERSION,
            "wafer_frontend.segmented_kv_state_transfer_contract/v1alpha1",
        )
        cases = (
            (2, 1, (0, 0, 0), (0, 2, 0)),
            (1, 2, (0, 2, 0), (0, 0, 0)),
        )
        for prefill_tp, decode_tp, source_offset, destination_offset in cases:
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                planned, logical, contracts = _case(prefill_tp, decode_tp)
                self.assertEqual(len(contracts), 8)
                self.assertEqual(sum(item.bytes for item in contracts), 1024)
                self.assertEqual(
                    tuple(item.logical_slice for item in contracts),
                    logical,
                )
                selected = contracts[2]
                self.assertEqual(
                    (
                        selected.logical_slice.source_local_offset,
                        selected.logical_slice.destination_local_offset,
                        selected.logical_slice.source_local_shape,
                        selected.bytes,
                        len(selected.segments),
                    ),
                    (source_offset, destination_offset, (8, 2, 4), 128, 8),
                )
                self.assertEqual(
                    selected.segments,
                    tuple(
                        KvTransferSegment(
                            source_local_offset=(token, source_offset[1], 0),
                            source_local_shape=(1, 2, 4),
                            destination_local_offset=(
                                token,
                                destination_offset[1],
                                0,
                            ),
                            destination_local_shape=(1, 2, 4),
                            bytes=16,
                        )
                        for token in range(8)
                    ),
                )
                self.assertEqual(
                    loads_dataclass(
                        SegmentedKvStateTransferContract,
                        canonical_json(selected),
                    ),
                    selected,
                )
                selected.validate_against(planned.graph)

    def test_segment_union_order_offsets_and_bytes_are_fail_closed(self) -> None:
        planned, _logical, contracts = _case(2, 1)
        contract = contracts[2]
        mutations = {
            "missing": contract.segments[:-1],
            "reordered": tuple(reversed(contract.segments)),
            "offset": (
                replace(
                    contract.segments[0],
                    destination_local_offset=(1, 2, 0),
                ),
                *contract.segments[1:],
            ),
            "bytes": (
                replace(contract.segments[0], bytes=17),
                replace(contract.segments[1], bytes=15),
                *contract.segments[2:],
            ),
        }
        for name, segments in mutations.items():
            with self.subTest(name=name):
                candidate = SegmentedKvStateTransferContract.create(
                    producer_pass=contract.producer_pass,
                    logical_slice=contract.logical_slice,
                    segments=segments,
                    bytes=sum(segment.bytes for segment in segments),
                )
                with self.assertRaisesRegex(
                    SchemaError,
                    "without gaps or overlap|original logical slice",
                ):
                    candidate.validate_against(planned.graph)

    def test_route_and_id_tamper_are_fail_closed(self) -> None:
        planned, _logical, contracts = _case(2, 1)
        contract = contracts[0]
        other_route = next(
            route
            for route in planned.graph.cross_routes
            if route.id != contract.logical_slice.cross_group_route_ref
        )
        logical_key = contract.logical_slice._semantic_key()
        logical_key["cross_group_route_ref"] = other_route.id
        wrong_route = SlicedKvStateTransferContract.create(
            producer_pass=contract.logical_slice.producer_pass,
            **logical_key,
        )
        candidate = SegmentedKvStateTransferContract.from_logical_slice(
            producer_pass=contract.producer_pass,
            logical_slice=wrong_route,
        )
        with self.assertRaisesRegex(SchemaError, "endpoints disagree"):
            candidate.validate_against(planned.graph)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(contract, id="segmented_kv_state_transfer_wrong").validate()


if __name__ == "__main__":
    unittest.main()
