from __future__ import annotations

from dataclasses import replace
import json
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    schedule_stage4 as public_schedule_stage4,
)
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_stage4,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_stage4
from llm.frontend.wafer_frontend.schema import (
    Stage4ScheduledIR2 as PublicStage4ScheduledIR2,
)
from llm.frontend.wafer_frontend.schema.ir2 import FlowRouteRole
from llm.frontend.wafer_frontend.schema.n5 import (
    STAGE4_SCHEDULED_IR2_SCHEMA_VERSION,
    Stage4ProjectToIR2Context,
    Stage4ScheduledIR2,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from _fixtures import naive_intra_die_scheduling_context
from test_stage4_carriers import _chain, _fused_chain


def _projected(*, fused: bool):
    planned = (_fused_chain() if fused else _chain(1, 1))[-1]
    projection_context = Stage4ProjectToIR2Context.create(
        producer_pass="stage4_scheduled_carrier_test"
    )
    return project_stage4(planned, projection_context)


def _projected_pdr():
    planned = _chain(2, 1)[-1]
    projection_context = Stage4ProjectToIR2Context.create(
        producer_pass="stage4_scheduled_carrier_test"
    )
    return project_stage4(planned, projection_context)


def _scheduling_context():
    return naive_intra_die_scheduling_context(
        "stage4_scheduled_carrier_test"
    )


class Stage4ScheduledCarrierTest(unittest.TestCase):
    def test_public_api_fused_tp1_capacity_and_strict_serde(self) -> None:
        self.assertIs(public_schedule_stage4, schedule_stage4)
        self.assertIs(PublicStage4ScheduledIR2, Stage4ScheduledIR2)
        source = _projected(fused=True)
        context = _scheduling_context()
        result = schedule_stage4(source, context)
        result.validate_against(source, context)
        self.assertEqual(
            STAGE4_SCHEDULED_IR2_SCHEMA_VERSION,
            "wafer_frontend.stage4_scheduled_ir2/v1alpha2",
        )
        self.assertEqual(
            [len(schedule.placements) for schedule in result.schedule_set.schedules],
            [92, 0],
        )
        self.assertEqual(
            [len(schedule.task_state_uses) for schedule in result.schedule_set.schedules],
            [42, 0],
        )
        self.assertEqual(result.projection.state_transfers, ())
        self.assertEqual(result.source_projected_carrier_id, source.id)
        self.assertEqual(result.source_planned_carrier_id, source.source_planned_carrier_id)
        self.assertEqual(result.projection_context_id, source.projection_context_id)
        self.assertEqual(result.scheduling_context_id, context.id)
        self.assertEqual(
            loads_dataclass(
                Stage4ScheduledIR2,
                canonical_json(result),
                path="result",
            ),
            result,
        )
        self.assertEqual(schedule_stage4(source, context), result)

        raw = json.loads(canonical_json(result))
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            loads_dataclass(
                Stage4ScheduledIR2,
                canonical_json(raw),
                path="result",
            )
        del raw["unexpected"]
        del raw["schedule_set"]
        with self.assertRaises(SchemaError):
            loads_dataclass(
                Stage4ScheduledIR2,
                canonical_json(raw),
                path="result",
            )

    def test_pds_tp1_capacity_and_cross_route_bindings_are_exact(self) -> None:
        source = _projected(fused=False)
        context = _scheduling_context()
        result = schedule_stage4(source, context)
        result.validate_against(source, context)
        schedules = result.schedule_set.schedules
        self.assertEqual(
            [len(schedule.placements) for schedule in schedules],
            [48, 52, 0],
        )
        self.assertEqual(
            [len(schedule.task_state_uses) for schedule in schedules],
            [19, 19, 0],
        )
        self.assertEqual(
            [len(schedule.flow_routes) for schedule in schedules],
            [4, 4, 0],
        )
        route_id = result.graph.cross_routes[0].id
        for schedule, dag, role in (
            (schedules[0], result.projection.dags[0], FlowRouteRole.SOURCE),
            (
                schedules[1],
                result.projection.dags[1],
                FlowRouteRole.DESTINATION,
            ),
        ):
            self.assertEqual(
                tuple(binding.flow_id for binding in schedule.flow_routes),
                tuple(flow.id for flow in dag.flows),
            )
            self.assertEqual(
                {binding.pair_route_ref for binding in schedule.flow_routes},
                {route_id},
            )
            self.assertEqual(
                {binding.role for binding in schedule.flow_routes},
                {role},
            )

    def test_pdr_tp2_to_tp1_segmented_capacity_and_routes_are_exact(self) -> None:
        source = _projected_pdr()
        context = _scheduling_context()
        result = schedule_stage4(source, context)
        result.validate_against(source, context)
        contracts = result.projection.state_transfers
        schedules = result.schedule_set.schedules
        self.assertEqual(len(contracts), 8)
        self.assertEqual(
            sum(len(contract.segments) for contract in contracts),
            64,
        )
        self.assertEqual(sum(contract.bytes for contract in contracts), 1024)
        self.assertEqual(len(result.graph.cross_routes), 2)
        self.assertEqual(
            tuple(
                (
                    schedule.die_id,
                    len(schedule.placements),
                    len(schedule.buffer_bindings),
                    len(schedule.flow_routes),
                )
                for schedule in schedules
            ),
            ((0, 112, 69, 48), (1, 112, 69, 80), (2, 172, 45, 64)),
        )
        self.assertEqual(result.source_projected_carrier_id, source.id)
        self.assertEqual(result.projection, source.projection)
        self.assertEqual(
            loads_dataclass(
                Stage4ScheduledIR2,
                canonical_json(result),
                path="result",
            ),
            result,
        )

    def test_version_source_context_projection_schedule_and_payload_tamper_reject(self) -> None:
        source = _projected(fused=False)
        context = _scheduling_context()
        result = schedule_stage4(source, context)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                result,
                schema_version="wafer_frontend.stage4_scheduled_ir2/v1alpha1",
            ).validate()
        fused_source = _projected(fused=True)
        with self.assertRaisesRegex(SchemaError, "source projected carrier"):
            result.validate_against(fused_source, context)
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            result.validate_against(source, replace(context, id="wrong"))
        for field_name, value in (
            ("source_planned_carrier_id", "wrong"),
            ("projection_context_id", "wrong"),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(SchemaError):
                    replace(result, **{field_name: value}).validate_against(
                        source,
                        context,
                    )
        with self.assertRaises(SchemaError):
            replace(
                result,
                projection=replace(result.projection, producer_pass="wrong"),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(
                result,
                schedule_set=replace(
                    result.schedule_set,
                    source_projection_id="wrong",
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "intra_die_schedule"):
            replace(
                result,
                schedule_set=replace(
                    result.schedule_set,
                    producer_pass="wrong",
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "intra_die_schedule"):
            replace(
                result,
                schedule_set=replace(
                    result.schedule_set,
                    schedules=(
                        replace(
                            result.schedule_set.schedules[0],
                            producer_pass="wrong",
                        ),
                        *result.schedule_set.schedules[1:],
                    ),
                ),
            ).validate()
        with self.assertRaises(SchemaError):
            replace(result, graph=fused_source.graph).validate_against(
                source,
                context,
            )


if __name__ == "__main__":
    unittest.main()
