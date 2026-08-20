from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes import (
    build_stage4_segmented_state_transfers,
    validate_stage4_segmented_state_transfers,
)
from llm.frontend.wafer_frontend.passes.stage4_state_transfer import (
    build_stage4_state_transfers,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    SegmentedKvStateTransferContract,
)

from test_stage4_carriers import _chain, _fused_chain


class Stage4SegmentedStateTransferPassTest(unittest.TestCase):
    def test_gather_and_scatter_sets_match_the_logical_oracle(self) -> None:
        for prefill_tp, decode_tp in ((2, 1), (1, 2)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                planned = _chain(prefill_tp, decode_tp)[-1]
                contracts = build_stage4_segmented_state_transfers(planned)
                validate_stage4_segmented_state_transfers(
                    contracts,
                    planned,
                )
                self.assertEqual(
                    contracts,
                    build_stage4_segmented_state_transfers(planned),
                )
                logical = build_stage4_state_transfers(
                    planned.graph,
                    planned.pd_plan,
                )
                self.assertEqual(
                    tuple(contract.logical_slice for contract in contracts),
                    logical,
                )
                self.assertEqual(
                    (
                        len(contracts),
                        {len(contract.segments) for contract in contracts},
                        {segment.bytes for contract in contracts for segment in contract.segments},
                        sum(contract.bytes for contract in contracts),
                    ),
                    (8, {8}, {16}, 1024),
                )
                self.assertEqual(
                    tuple(
                        segment.source_local_offset[0]
                        for segment in contracts[0].segments
                    ),
                    tuple(range(8)),
                )
                self.assertEqual(
                    tuple(
                        segment.destination_local_offset[0]
                        for segment in contracts[0].segments
                    ),
                    tuple(range(8)),
                )

    def test_complete_set_order_producer_and_segments_fail_closed(self) -> None:
        planned = _chain(2, 1)[-1]
        contracts = build_stage4_segmented_state_transfers(planned)
        with self.assertRaisesRegex(
            SchemaError,
            "planned segmented KV transfer set",
        ):
            validate_stage4_segmented_state_transfers(
                tuple(reversed(contracts)),
                planned,
            )
        wrong_producer = (
            replace(contracts[0], producer_pass="forged"),
            *contracts[1:],
        )
        with self.assertRaisesRegex(
            SchemaError,
            "stage4_segmented_state_transfer",
        ):
            validate_stage4_segmented_state_transfers(
                wrong_producer,
                planned,
            )
        first = contracts[0]
        wrong_segments = SegmentedKvStateTransferContract.create(
            producer_pass=first.producer_pass,
            logical_slice=first.logical_slice,
            segments=tuple(reversed(first.segments)),
            bytes=first.bytes,
        )
        with self.assertRaisesRegex(SchemaError, "without gaps or overlap"):
            validate_stage4_segmented_state_transfers(
                (wrong_segments, *contracts[1:]),
                planned,
            )

    def test_non_segmented_modes_are_not_accepted(self) -> None:
        for planned in (_chain(1, 1)[-1], _fused_chain()[-1]):
            with self.subTest(mode=planned.pd_plan.mode):
                with self.assertRaisesRegex(
                    UnsupportedFeatureError,
                    "segmented KV transfers require",
                ):
                    build_stage4_segmented_state_transfers(planned)


if __name__ == "__main__":
    unittest.main()
