from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import (
    SchemaError,
    UnsupportedFeatureError,
)
from llm.frontend.wafer_frontend.passes import (
    build_stage4_planned_state_transfers,
    build_stage4_state_transfers,
    place_stage4_ir0,
    validate_stage4_planned_state_transfers,
    validate_stage4_state_transfers,
)
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.passes.stage4_pd import build_stage4_pd_plan
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind

from test_stage4_carriers import _chain, _fused_chain
from test_stage4_pd import _profile, _spec
from test_stage4_placement import _tp1_case


class Stage4StateTransferPassTest(unittest.TestCase):
    def test_tp1_real_plan_placement_and_kv_accesses_close_exactly(self) -> None:
        graph, context, plan = _tp1_case()
        ir1 = place_stage4_ir0(graph, context, plan)

        contracts = build_stage4_state_transfers(ir1, plan)
        validate_stage4_state_transfers(contracts, ir1, plan)
        self.assertEqual(contracts, build_stage4_state_transfers(ir1, plan))
        self.assertEqual(len(contracts), 4)
        self.assertEqual([contract.bytes for contract in contracts], [256] * 4)
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(
            {contract.cross_group_route_ref for contract in contracts},
            {ir1.cross_routes[0].id},
        )

        assert ir1.persistent_state_manifest is not None
        declarations = {
            declaration.id: declaration
            for declaration in ir1.persistent_state_manifest.declarations
        }
        accesses = {access.id: access for access in ir1.state_accesses}
        observed = []
        for contract in contracts:
            source = declarations[
                accesses[contract.source_state_access_ref].state_ref
            ].identity
            destination = declarations[
                accesses[contract.destination_state_access_ref].state_ref
            ].identity
            observed.append(
                (
                    source.layer_index,
                    source.kind,
                    source.instance_ref,
                    destination.instance_ref,
                    contract.source_local_offset,
                    contract.source_local_shape,
                    contract.destination_local_offset,
                    contract.destination_local_shape,
                )
            )
        self.assertEqual(
            observed,
            [
                (layer, kind, "P0", "D0", (0, 0, 0), (8, 4, 4),
                 (0, 0, 0), (8, 4, 4))
                for layer in range(2)
                for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            ],
        )

    def test_planned_carrier_api_is_canonical_and_complete(self) -> None:
        planned = _chain(1, 1)[-1]

        contracts = build_stage4_planned_state_transfers(planned)
        validate_stage4_planned_state_transfers(contracts, planned)
        self.assertEqual(
            contracts,
            build_stage4_state_transfers(planned.graph, planned.pd_plan),
        )
        self.assertEqual(
            (
                len(contracts),
                tuple(contract.bytes for contract in contracts),
                sum(contract.bytes for contract in contracts),
            ),
            (4, (256, 256, 256, 256), 1024),
        )
        assert planned.graph.persistent_state_manifest is not None
        declarations = {
            declaration.id: declaration
            for declaration in planned.graph.persistent_state_manifest.declarations
        }
        accesses = {
            access.id: access for access in planned.graph.state_accesses
        }
        self.assertEqual(
            tuple(
                declarations[
                    accesses[contract.source_state_access_ref].state_ref
                ].identity.kind
                for contract in contracts
            ),
            (
                StateKind.KV_KEY,
                StateKind.KV_VALUE,
                StateKind.KV_KEY,
                StateKind.KV_VALUE,
            ),
        )
        with self.assertRaisesRegex(
            SchemaError,
            "plan/IR1-derived sliced KV transfers",
        ):
            validate_stage4_planned_state_transfers(
                contracts[:-1],
                planned,
            )
        with self.assertRaisesRegex(
            SchemaError,
            "Stage4InterDiePlannedIR1",
        ):
            build_stage4_planned_state_transfers(planned.graph)  # type: ignore[arg-type]

    def test_planned_fused_is_exactly_empty(self) -> None:
        planned = _fused_chain()[-1]
        self.assertEqual(build_stage4_planned_state_transfers(planned), ())
        validate_stage4_planned_state_transfers((), planned)
        separated = _chain(1, 1)[-1]
        nonempty = build_stage4_planned_state_transfers(separated)
        with self.assertRaisesRegex(
            SchemaError,
            "fused PD must have an empty state-transfer set",
        ):
            validate_stage4_planned_state_transfers(nonempty, planned)

    def test_planned_segmented_reshard_is_explicitly_unsupported(self) -> None:
        for prefill_tp, decode_tp in ((2, 1), (1, 2)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                planned = _chain(prefill_tp, decode_tp)[-1]
                with self.assertRaisesRegex(
                    UnsupportedFeatureError,
                    "segmented KV reshard state transfers",
                ):
                    build_stage4_planned_state_transfers(planned)

    def test_complete_set_order_and_producer_are_fail_closed(self) -> None:
        graph, context, plan = _tp1_case()
        ir1 = place_stage4_ir0(graph, context, plan)
        contracts = build_stage4_state_transfers(ir1, plan)

        with self.assertRaisesRegex(
            SchemaError,
            "plan/IR1-derived sliced KV transfers",
        ):
            validate_stage4_state_transfers(
                tuple(reversed(contracts)),
                ir1,
                plan,
            )
        wrong_producer = (
            replace(contracts[0], producer_pass="forged"),
            *contracts[1:],
        )
        with self.assertRaisesRegex(SchemaError, "stage4_state_transfer"):
            validate_stage4_state_transfers(wrong_producer, ir1, plan)

    def test_missing_routes_and_fused_plan_are_rejected(self) -> None:
        graph, context, plan = _tp1_case()
        generic = place_ir0(graph, context)
        with self.assertRaisesRegex(
            SchemaError,
            "flow endpoint pairs",
        ):
            build_stage4_state_transfers(generic, plan)

        fused_spec = _spec(1, 1, fused=True)
        fused_plan = build_stage4_pd_plan(
            fused_spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError,
            "requires separated PD",
        ):
            build_stage4_state_transfers(generic, fused_plan)


if __name__ == "__main__":
    unittest.main()
