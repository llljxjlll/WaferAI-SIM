from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_moe_n6 import (
    build_lite_moe_n6_intent,
    validate_lite_moe_n6_intent,
)
from llm.frontend.wafer_frontend.schema.lite_moe_execution import LiteMoeTaskKind
from llm.frontend.wafer_frontend.schema.lite_moe_n6 import LiteMoeN6Intent
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from test_lite_moe_execution import LiteMoeExecutionTest


class LiteMoeN6IntentTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture = LiteMoeExecutionTest()
        fixture.setUp()
        self.n4 = fixture.n4
        self.projection = fixture.projection
        self.schedule = fixture.schedule
        self.global_dag = fixture.global_dag
        self.intent = build_lite_moe_n6_intent(
            self.global_dag,
            self.schedule,
            self.projection,
            self.n4,
        )

    def test_exact_buffer_state_compute_and_dte_closure(self) -> None:
        self.assertEqual(
            (
                len(self.intent.buffer_abis),
                len(self.intent.state_loads),
                len(self.intent.compute_units),
                len(self.intent.dte_units),
            ),
            (64, 24, 32, 8),
        )
        for abi in self.intent.buffer_abis:
            abi.validate("buffer_abi")
            self.assertEqual(abi.alignment_bytes, 64)
            self.assertLessEqual(
                abi.region_offset_bytes + abi.size_bytes,
                65536,
            )
        self.assertEqual(
            Counter(unit.kind for unit in self.intent.compute_units),
            Counter({LiteMoeTaskKind.GEMM: 24, LiteMoeTaskKind.SWIGLU: 8}),
        )
        by_node = {unit.node_ref: unit for unit in self.intent.compute_units}
        for swiglu in (
            unit
            for unit in self.intent.compute_units
            if unit.kind is LiteMoeTaskKind.SWIGLU
        ):
            prefix = swiglu.node_ref.rsplit(".", 1)[0]
            gate = by_node[f"{prefix}.gate"]
            up = by_node[f"{prefix}.up"]
            self.assertEqual(
                (gate.output.buffer_abi_ref, up.output.buffer_abi_ref),
                (swiglu.inputs[0].buffer_abi_ref,) * 2,
            )
            self.assertEqual(
                (
                    gate.output.offset_bytes,
                    gate.output.size_bytes,
                    up.output.offset_bytes,
                    up.output.size_bytes,
                    swiglu.inputs[0].offset_bytes,
                    swiglu.inputs[0].size_bytes,
                ),
                (0, 64, 64, 64, 0, 128),
            )
        action_ids = {item.id for item in self.global_dag.actions}
        claimed = {
            unit.action_ref for unit in self.intent.state_loads
        } | {
            unit.action_ref for unit in self.intent.compute_units
        } | {
            action
            for unit in self.intent.dte_units
            for action in (
                unit.send_action_ref,
                unit.recv_action_ref,
                unit.wait_action_ref,
            )
        }
        self.assertEqual(claimed, action_ids)
        self.assertEqual(
            sum(unit.bytes for unit in self.intent.dte_units), 256
        )
        self.assertEqual(
            len({unit.channel_symbol for unit in self.intent.dte_units}), 8
        )
        self.assertEqual(
            len({unit.token_symbol for unit in self.intent.dte_units}), 8
        )

    def test_deterministic_strict_round_trip_and_old_version_rejected(self) -> None:
        self.assertEqual(
            canonical_digest(
                build_lite_moe_n6_intent(
                    self.global_dag,
                    self.schedule,
                    self.projection,
                    self.n4,
                )
            ),
            canonical_digest(self.intent),
        )
        self.assertEqual(
            loads_dataclass(
                LiteMoeN6Intent,
                canonical_json(self.intent),
                path="intent",
            ),
            self.intent,
        )
        with self.assertRaisesRegex(SchemaError, "unsupported"):
            replace(self.intent, schema_version="old").validate()

    def test_compute_state_dte_and_provenance_tamper_fail_closed(self) -> None:
        up_index = next(
            index
            for index, unit in enumerate(self.intent.compute_units)
            if unit.node_ref.endswith(".up")
        )
        up = self.intent.compute_units[up_index]
        wrong_compute = replace(
            up,
            output=replace(up.output, offset_bytes=32),
        )
        forged = LiteMoeN6Intent.create(
            **(
                self.intent._semantic_key()
                | {
                    "compute_units": self.intent.compute_units[:up_index]
                    + (wrong_compute,)
                    + self.intent.compute_units[up_index + 1 :]
                }
            )
        )
        with self.assertRaisesRegex(SchemaError, "exact lowering quotient"):
            validate_lite_moe_n6_intent(
                forged,
                self.global_dag,
                self.schedule,
                self.projection,
                self.n4,
            )

        first_dte, second_dte = self.intent.dte_units[:2]
        with self.assertRaisesRegex(SchemaError, "unique"):
            LiteMoeN6Intent.create(
                **(
                    self.intent._semantic_key()
                    | {
                        "dte_units": (
                            first_dte,
                            replace(
                                second_dte,
                                token_symbol=first_dte.token_symbol,
                            ),
                            *self.intent.dte_units[2:],
                        )
                    }
                )
            )
        wrong_source = LiteMoeN6Intent.create(
            **(
                self.intent._semantic_key()
                | {"source_global_id": "forged"}
            )
        )
        with self.assertRaisesRegex(SchemaError, "exact lowering quotient"):
            validate_lite_moe_n6_intent(
                wrong_source,
                self.global_dag,
                self.schedule,
                self.projection,
                self.n4,
            )

        with self.assertRaisesRegex(SchemaError, "exact 24 state"):
            LiteMoeN6Intent.create(
                **(
                    self.intent._semantic_key()
                    | {"state_loads": self.intent.state_loads[1:]}
                )
            )


if __name__ == "__main__":
    unittest.main()
