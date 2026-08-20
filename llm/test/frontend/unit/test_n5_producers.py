from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend import passes
from llm.frontend.wafer_frontend.passes.global_action_dag import (
    build_global_bundle,
    build_global_profile,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import (
    project_bundle,
    project_profile,
)
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_bundle,
    schedule_profile,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_n5_schema import (
    _multi_profile_fixture,
    _projection_for_graph,
    _schedule_for_projection,
)
from test_global_action_schema import _create_global


class RecordingProjector:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(
        self, ir1, fusion_plans, standalone_plans, *, state_transfers
    ):
        self.assert_no_transfers(state_transfers)
        self.calls.append(ir1.profile.stable_id())
        return _projection_for_graph(ir1)

    @staticmethod
    def assert_no_transfers(state_transfers) -> None:
        if state_transfers != ():
            raise AssertionError("fixture expects no state transfers")


class RecordingScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def schedule(self, projection, ir1):
        self.calls.append(ir1.profile.stable_id())
        return _schedule_for_projection(ir1, projection)


class RecordingGlobalBuilder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def build(self, ir1, projection, schedule_set):
        self.calls.append(ir1.profile.stable_id())
        return replace(
            _create_global(ir1, projection, schedule_set),
            producer_pass="global_action_dag",
        )


class ProjectToIR2ProducerTest(unittest.TestCase):
    def test_all_n5_producers_are_public_pass_exports(self) -> None:
        self.assertIs(passes.project_profile, project_profile)
        self.assertIs(passes.project_bundle, project_bundle)
        self.assertIs(passes.schedule_profile, schedule_profile)
        self.assertIs(passes.schedule_bundle, schedule_bundle)
        self.assertIs(passes.build_global_profile, build_global_profile)
        self.assertIs(passes.build_global_bundle, build_global_bundle)

    def test_profile_and_bundle_call_once_in_source_order(self) -> None:
        planned, context, _projected, *_rest = _multi_profile_fixture()
        projector = RecordingProjector()
        first = project_profile(planned.entries[0], context, projector)
        first.validate_against(planned.entries[0], context)
        self.assertEqual(projector.calls, [planned.entries[0].profile_id])

        projector = RecordingProjector()
        result = project_bundle(planned, context, projector)
        result.validate_against(planned, context)
        self.assertEqual(
            projector.calls,
            [entry.profile_id for entry in planned.entries],
        )
        self.assertEqual(
            tuple(entry.source_planned_entry_id for entry in result.entries),
            tuple(entry.id for entry in planned.entries),
        )

    def test_input_is_immutable_and_default_is_deterministic(self) -> None:
        planned, context, _projected, *_rest = _multi_profile_fixture()
        before = canonical_digest((planned, context))
        first = project_bundle(planned, context)
        second = project_bundle(planned, context)
        self.assertEqual(canonical_digest((planned, context)), before)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(first), canonical_digest(second))

    def test_wrong_output_type_producer_and_source_are_rejected(self) -> None:
        planned, context, _projected, *_rest = _multi_profile_fixture()

        class WrongType:
            def run(
                self, ir1, fusion_plans, standalone_plans, *, state_transfers
            ):
                return object()

        with self.assertRaisesRegex(SchemaError, "IR2ProjectionResult"):
            project_profile(planned.entries[0], context, WrongType())

        class WrongProducer:
            def run(
                self, ir1, fusion_plans, standalone_plans, *, state_transfers
            ):
                return replace(
                    _projection_for_graph(ir1),
                    producer_pass="impostor",
                )

        with self.assertRaisesRegex(SchemaError, "project_to_ir2"):
            project_profile(planned.entries[0], context, WrongProducer())

        first_projection = _projection_for_graph(planned.entries[0].graph)

        class WrongSource:
            def run(
                self, ir1, fusion_plans, standalone_plans, *, state_transfers
            ):
                return first_projection

        with self.assertRaisesRegex(SchemaError, "different IR-1"):
            project_profile(planned.entries[1], context, WrongSource())


class IntraDieScheduleProducerTest(unittest.TestCase):
    def test_profile_and_bundle_call_once_in_source_order(self) -> None:
        (
            _planned,
            _projection_context,
            projected,
            context,
            _scheduled,
            _global,
        ) = _multi_profile_fixture()
        policy = RecordingScheduler()
        first = schedule_profile(projected.entries[0], context, policy)
        first.validate_against(projected.entries[0], context)
        self.assertEqual(policy.calls, [projected.entries[0].profile_id])

        policy = RecordingScheduler()
        result = schedule_bundle(projected, context, policy)
        result.validate_against(projected, context)
        self.assertEqual(
            policy.calls,
            [entry.profile_id for entry in projected.entries],
        )
        self.assertEqual(
            tuple(entry.source_projected_entry_id for entry in result.entries),
            tuple(entry.id for entry in projected.entries),
        )

    def test_injected_policy_is_immutable_and_deterministic(self) -> None:
        (
            _planned,
            _projection_context,
            projected,
            context,
            _scheduled,
            _global,
        ) = _multi_profile_fixture()
        before = canonical_digest((projected, context))
        first = schedule_bundle(projected, context, RecordingScheduler())
        second = schedule_bundle(projected, context, RecordingScheduler())
        self.assertEqual(canonical_digest((projected, context)), before)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(first), canonical_digest(second))

    def test_wrong_output_type_producer_and_source_are_rejected(self) -> None:
        (
            _planned,
            _projection_context,
            projected,
            context,
            _scheduled,
            _global,
        ) = _multi_profile_fixture()

        class WrongType:
            def schedule(self, projection, ir1):
                return object()

        with self.assertRaisesRegex(SchemaError, "IntraDieScheduleSet"):
            schedule_profile(projected.entries[0], context, WrongType())

        class WrongProducer:
            def schedule(self, projection, ir1):
                return replace(
                    _schedule_for_projection(ir1, projection),
                    producer_pass="impostor",
                )

        with self.assertRaisesRegex(SchemaError, "intra_die_schedule"):
            schedule_profile(projected.entries[0], context, WrongProducer())

        first_schedule = _schedule_for_projection(
            projected.entries[0].graph,
            projected.entries[0].projection,
        )

        class WrongSource:
            def schedule(self, projection, ir1):
                return first_schedule

        with self.assertRaisesRegex(SchemaError, "DAG order"):
            schedule_profile(projected.entries[1], context, WrongSource())


class GlobalActionProducerTest(unittest.TestCase):
    def test_profile_and_bundle_call_once_in_source_order(self) -> None:
        *_, scheduled, _global = _multi_profile_fixture()
        builder = RecordingGlobalBuilder()
        first = build_global_profile(scheduled.entries[0], builder)
        first.validate_against(scheduled.entries[0])
        self.assertEqual(builder.calls, [scheduled.entries[0].profile_id])

        builder = RecordingGlobalBuilder()
        result = build_global_bundle(scheduled, builder)
        result.validate_against(scheduled)
        self.assertEqual(
            builder.calls,
            [entry.profile_id for entry in scheduled.entries],
        )
        self.assertEqual(
            tuple(entry.source_scheduled_entry_id for entry in result.entries),
            tuple(entry.id for entry in scheduled.entries),
        )

    def test_input_is_immutable_and_default_is_deterministic(self) -> None:
        *_, scheduled, _global = _multi_profile_fixture()
        before = canonical_digest(scheduled)
        first = build_global_bundle(scheduled)
        second = build_global_bundle(scheduled)
        self.assertEqual(canonical_digest(scheduled), before)
        self.assertEqual(first, second)
        self.assertEqual(canonical_digest(first), canonical_digest(second))
        self.assertTrue(
            all(entry.lowering_context() for entry in first.entries)
        )

    def test_wrong_output_type_producer_and_source_are_rejected(self) -> None:
        *_, scheduled, _global = _multi_profile_fixture()

        class WrongType:
            def build(self, ir1, projection, schedule_set):
                return object()

        with self.assertRaisesRegex(SchemaError, "GlobalActionDAG"):
            build_global_profile(scheduled.entries[0], WrongType())

        class WrongProducer:
            def build(self, ir1, projection, schedule_set):
                return _create_global(ir1, projection, schedule_set)

        with self.assertRaisesRegex(SchemaError, "global_action_dag"):
            build_global_profile(scheduled.entries[0], WrongProducer())

        first_global = replace(
            _create_global(
                scheduled.entries[0].graph,
                scheduled.entries[0].projection,
                scheduled.entries[0].schedule_set,
            ),
            producer_pass="global_action_dag",
        )

        class WrongSource:
            def build(self, ir1, projection, schedule_set):
                return first_global

        with self.assertRaisesRegex(SchemaError, "different IR-1"):
            build_global_profile(scheduled.entries[1], WrongSource())


if __name__ == "__main__":
    unittest.main()
