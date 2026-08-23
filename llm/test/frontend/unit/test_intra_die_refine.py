from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.passes.intra_die_refine import (
    refine_bundle,
    refine_profile,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    IntraDieRefineContext,
    IntraDieRefineContract,
    RefinedProfileIR2,
)
from llm.frontend.wafer_frontend.schema.local_transport import (
    LocalEvent,
    LocalEventPhase,
    LocalFlow,
    LocalNocRoute,
    LocalTransportPlan,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir2 import TensorSlice
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_n5_schema import _projected_fixture


class IntraDieRefineTest(unittest.TestCase):
    def _context(self) -> IntraDieRefineContext:
        selection = production_registry().instantiate(
            RegistryKind.INTRA_DIE, "naive"
        ).selection
        return IntraDieRefineContext.create(
            producer_pass="test_intra_die_refine",
            policy=selection,
        )

    def test_identity_profile_preserves_exact_projection(self) -> None:
        _planned, _projection_context, projected = _projected_fixture()
        context = self._context()
        before = canonical_digest((projected, context))
        refined = refine_profile(projected.entries[0], context)
        refined.validate_against(projected.entries[0], context)
        self.assertIs(refined.projection, projected.entries[0].projection)
        self.assertEqual(canonical_digest((projected, context)), before)

    def test_identity_bundle_is_deterministic_and_canonical(self) -> None:
        _planned, _projection_context, projected = _projected_fixture()
        context = self._context()
        first = refine_bundle(projected, context)
        second = refine_bundle(projected, context)
        first.validate_against(projected, context)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(entry.source.id for entry in first.entries),
            tuple(entry.id for entry in projected.entries),
        )

    def test_identity_wrapper_rejects_changed_projection(self) -> None:
        _planned, _projection_context, projected = _projected_fixture()
        context = self._context()
        with self.assertRaisesRegex(Exception, "IR2ProjectionResult"):
            RefinedProfileIR2.create(
                source=projected.entries[0],
                context=context,
                projection=object(),  # type: ignore[arg-type]
            ).validate()

    def test_local_transport_contract_requires_and_carries_one_plan(self) -> None:
        _planned, _projection_context, projected = _projected_fixture()
        source = projected.entries[0]
        selection = production_registry().instantiate(RegistryKind.INTRA_DIE, "naive").selection
        context = IntraDieRefineContext.create(
            producer_pass="test_intra_die_refine",
            policy=selection,
            contract=IntraDieRefineContract.LOCAL_TRANSPORT_V1,
        )
        flow = LocalFlow(
            id="local_0", source_task_id="producer", destination_task_id="consumer",
            source_core_id=3, destination_core_id=7, value_id="value_0",
            tensor_slice=TensorSlice("value_0", (0,), (16,)), bytes=32,
            dtype=DType.FP16, route_id="route_0", send_event_id="event_send_0",
            recv_event_id="event_recv_0",
        )
        plan = LocalTransportPlan.create(
            producer_pass="test_intra_die_refine", source_dag_id=source.projection.dags[0].id,
            flows=(flow,), routes=(LocalNocRoute("route_0", "local_0", ((0, 0), (1, 0), (1, 1))),),
            events=(
                LocalEvent("event_recv_0", "local_0", "consumer", LocalEventPhase.RECV_READY),
                LocalEvent("event_send_0", "local_0", "producer", LocalEventPhase.SEND_COMPLETE),
            ),
        )
        refined = refine_profile(source, context, local_transport_plan=plan)
        self.assertEqual(refined.local_transport_plan, plan)
        with self.assertRaisesRegex(Exception, "requires a local transport plan"):
            refine_profile(source, context)


if __name__ == "__main__":
    unittest.main()
